//! Candle-based real SLM inference backend for NRT.
//!
//! Loads quantized GGUF weights from HuggingFace and runs real forward passes
//! through the NRT `Backend` trait. No FFI, no Python — pure Rust end-to-end.
//!
//! The backend spawns CPU work on `tokio::task::spawn_blocking` because Candle's
//! forward pass is synchronous and CPU/Metal-bound. Each model is protected by
//! a `std::sync::Mutex` (not tokio's) because the lock is held for the entire
//! forward pass.
//!
//! Each `ModelId` is backed by a `ModelProfile` describing which GGUF file to
//! fetch, which tokenizer to use, and optional intent labels the router may
//! emit. The mapping is supplied at `CandleBackend::register` time so multiple
//! tiny specialists can share the same underlying weights with different
//! system prompts — a useful pattern for prototyping ensembles without
//! downloading four separate models.

mod model_map;
mod profile;
mod session;

pub use model_map::ModelMap;
pub use profile::{
    default_tinyllama_profile, router_profile, specialist_profile, ModelProfile, PromptHeader,
    RouterConfig,
};

use async_trait::async_trait;
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use nrt_core::{
    Backend, BackendLoadHandle, InferenceRequest, InferenceResponse, KvCacheHandle, ModelId,
    NrtError, NrtResult, Tier,
};
use parking_lot::RwLock;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::Instant,
};
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument, warn};

#[derive(Debug, Error)]
pub enum CandleError {
    #[error("candle core error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("hub fetch error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("model {0} not registered")]
    UnknownModel(String),
    #[error("model {0} not loaded")]
    NotLoaded(String),
    #[error("generation produced no tokens")]
    EmptyGeneration,
}

impl From<CandleError> for NrtError {
    fn from(e: CandleError) -> Self {
        NrtError::backend(e.to_string())
    }
}

/// Held inside the backend for each loaded model. The `StdMutex` is held for
/// the duration of a forward pass — fine because inference is the unit of work.
struct LoadedModel {
    weights: StdMutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    /// Monotonic position for kv-cache indexing on the quantized_llama model.
    kv_pos: StdMutex<usize>,
    profile: ModelProfile,
}

pub struct CandleBackend {
    device: Device,
    profiles: RwLock<ModelMap>,
    loaded: RwLock<std::collections::HashMap<ModelId, Arc<LoadedModel>>>,
    load_counter: AtomicU64,
}

impl CandleBackend {
    /// Build a backend using Metal on macOS if available, CPU otherwise.
    /// Callers pass a `ModelMap` describing the registered profiles.
    pub fn new(profiles: ModelMap) -> Result<Self, CandleError> {
        let device = pick_device();
        info!(target: "candle", device = ?device, "CandleBackend starting");
        Ok(Self {
            device,
            profiles: RwLock::new(profiles),
            loaded: RwLock::new(Default::default()),
            load_counter: AtomicU64::new(0),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn register(&self, id: ModelId, profile: ModelProfile) {
        self.profiles.write().insert(id, profile);
    }

    fn profile_for(&self, id: &ModelId) -> Result<ModelProfile, CandleError> {
        self.profiles
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| CandleError::UnknownModel(id.as_str().to_string()))
    }

    fn loaded_for(&self, id: &ModelId) -> Result<Arc<LoadedModel>, CandleError> {
        self.loaded
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| CandleError::NotLoaded(id.as_str().to_string()))
    }

    #[instrument(skip(self), fields(model = %model))]
    async fn load_weights(&self, model: &ModelId) -> Result<Arc<LoadedModel>, CandleError> {
        if let Some(m) = self.loaded.read().get(model).cloned() {
            return Ok(m);
        }
        let profile = self.profile_for(model)?;
        let device = self.device.clone();
        let model_for_blocking = model.clone();

        let loaded = tokio::task::spawn_blocking(move || -> Result<Arc<LoadedModel>, CandleError> {
            let api = hf_hub::api::sync::Api::new()
                .map_err(CandleError::Hub)?;
            let gguf_repo = api.repo(hf_hub::Repo::new(
                profile.gguf_repo.clone(),
                hf_hub::RepoType::Model,
            ));
            let gguf_path: PathBuf = gguf_repo
                .get(&profile.gguf_file)
                .map_err(CandleError::Hub)?;
            info!(
                target: "candle",
                model = %model_for_blocking,
                path = %gguf_path.display(),
                "gguf fetched"
            );

            let mut file = std::fs::File::open(&gguf_path)?;
            let gguf = gguf_file::Content::read(&mut file)?;
            let weights = ModelWeights::from_gguf(gguf, &mut file, &device)?;

            // Fetch tokenizer via the tokenizers crate's own downloader.
            // hf-hub 0.3 has URL-base parsing quirks with some repos; using
            // `Tokenizer::from_pretrained` here sidesteps them entirely.
            let tokenizer = Tokenizer::from_pretrained(&profile.tokenizer_repo, None)
                .map_err(|e| CandleError::Tokenizer(format!(
                    "from_pretrained({:?}): {e}", profile.tokenizer_repo
                )))?;

            Ok(Arc::new(LoadedModel {
                weights: StdMutex::new(weights),
                tokenizer,
                device: device.clone(),
                kv_pos: StdMutex::new(0),
                profile,
            }))
        })
        .await
        .map_err(|e| CandleError::Tokenizer(format!("spawn_blocking join: {e}")))??;

        self.loaded.write().insert(model.clone(), loaded.clone());
        Ok(loaded)
    }

    fn next_token(&self) -> u64 {
        self.load_counter.fetch_add(1, Ordering::Relaxed)
    }
}

fn pick_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

#[async_trait]
impl Backend for CandleBackend {
    fn name(&self) -> &'static str {
        "candle"
    }

    async fn load(&self, model: &ModelId, tier: Tier) -> NrtResult<BackendLoadHandle> {
        // Remote tier: no local weights, no download. Inference returns a
        // canned response indicating where the real call would go (e.g. an
        // upstream Claude endpoint). This matches the spec's fallback shape.
        if matches!(tier, Tier::Remote) {
            return Ok(BackendLoadHandle {
                model_id: model.clone(),
                resident_vram_mb: 0,
                standby_ram_mb: 0,
                load_token: self.next_token(),
                tier,
            });
        }
        let loaded = self.load_weights(model).await?;
        let resident_vram_mb = match tier {
            Tier::Resident | Tier::Active => loaded.profile.nominal_vram_mb,
            _ => 0,
        };
        let standby_ram_mb = match tier {
            Tier::Standby => loaded.profile.nominal_ram_mb,
            Tier::Resident | Tier::Active => 0,
            Tier::Remote => 0,
        };
        let handle = BackendLoadHandle {
            model_id: model.clone(),
            resident_vram_mb,
            standby_ram_mb,
            load_token: self.next_token(),
            tier,
        };
        Ok(handle)
    }

    async fn promote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()> {
        if handle.tier == to {
            return Ok(());
        }
        // Candle holds weights in memory always (CPU or Metal unified memory);
        // promotion is a bookkeeping move, not a data move. We still simulate a
        // small cost for the unified-memory pin so the metrics reflect reality.
        if matches!((handle.tier, to), (Tier::Standby, Tier::Resident)) {
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
            let profile = self.profile_for(&handle.model_id)?;
            handle.resident_vram_mb = profile.nominal_vram_mb;
            handle.standby_ram_mb = 0;
            handle.tier = Tier::Resident;
            return Ok(());
        }
        if matches!((handle.tier, to), (Tier::Resident, Tier::Active)) {
            handle.tier = Tier::Active;
            return Ok(());
        }
        Err(NrtError::InvalidTransition {
            from: handle.tier,
            to,
            reason: "candle backend supports Standby->Resident->Active path".into(),
        })
    }

    async fn demote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()> {
        if handle.tier == to {
            return Ok(());
        }
        match (handle.tier, to) {
            (Tier::Active, Tier::Resident) => {
                handle.tier = Tier::Resident;
                Ok(())
            }
            (Tier::Resident, Tier::Standby) => {
                let profile = self.profile_for(&handle.model_id)?;
                handle.resident_vram_mb = 0;
                handle.standby_ram_mb = profile.nominal_ram_mb;
                handle.tier = Tier::Standby;
                Ok(())
            }
            (from, to) => Err(NrtError::InvalidTransition {
                from,
                to,
                reason: "candle backend demote path not supported".into(),
            }),
        }
    }

    async fn unload(&self, handle: &BackendLoadHandle) -> NrtResult<()> {
        self.loaded.write().remove(&handle.model_id);
        Ok(())
    }

    async fn infer(
        &self,
        handle: &BackendLoadHandle,
        _kv: Option<&KvCacheHandle>,
        req: InferenceRequest,
    ) -> NrtResult<InferenceResponse> {
        if matches!(handle.tier, Tier::Remote) {
            return Ok(InferenceResponse {
                session_id: req.session_id,
                model_id: req.model_id,
                completion: format!(
                    "[remote-fallback] would dispatch to {}",
                    handle.model_id
                ),
                tokens_emitted: 0,
                intent: None,
                latency_ms: 1,
            });
        }
        if !matches!(handle.tier, Tier::Resident | Tier::Active) {
            return Err(NrtError::backend(format!(
                "candle cannot infer in tier {:?}",
                handle.tier
            )));
        }
        let loaded = self.loaded_for(&handle.model_id)?;
        let prompt = req.prompt.clone();
        let session_id = req.session_id;
        let model_id = req.model_id.clone();
        let max_tokens = req.max_tokens.min(128).max(1);

        let t0 = Instant::now();
        let (completion, intent, tokens_emitted) = tokio::task::spawn_blocking(move || {
            session::generate(&*loaded, &prompt, max_tokens as usize)
        })
        .await
        .map_err(|e| NrtError::backend(format!("spawn_blocking join: {e}")))??;
        let latency_ms = t0.elapsed().as_millis() as u64;
        debug!(
            target: "candle",
            model = %handle.model_id,
            latency_ms,
            tokens_emitted,
            "candle forward complete"
        );

        Ok(InferenceResponse {
            session_id,
            model_id,
            completion,
            tokens_emitted,
            intent,
            latency_ms,
        })
    }
}

impl CandleBackend {
    /// Pre-populate the `loaded` map by fetching weights for every registered
    /// model ahead of cluster bootstrap. Useful for the demo server so the
    /// first `/v1/chat/completions` call doesn't pay the download cost.
    pub async fn warm_all_registered(&self) -> Result<Vec<ModelId>, CandleError> {
        let ids: Vec<ModelId> = self.profiles.read().ids();
        let mut warmed = Vec::with_capacity(ids.len());
        for id in &ids {
            match self.load_weights(id).await {
                Ok(_) => {
                    warmed.push(id.clone());
                }
                Err(e) => {
                    warn!(target: "candle", model = %id, error = %e, "prewarm failed");
                    return Err(e);
                }
            }
        }
        Ok(warmed)
    }

    /// Register a lightweight profile for Remote-tier models so `load_weights`
    /// is never called on them. The profile is required by other parts of the
    /// backend (tier accounting) but weights are never fetched.
    pub fn register_remote(&self, id: ModelId) {
        let mut profile = default_tinyllama_profile();
        profile.nominal_vram_mb = 0;
        profile.nominal_ram_mb = 0;
        self.profiles.write().insert(id, profile);
    }
}

impl session::InferenceHost for LoadedModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn weights(&self) -> &StdMutex<ModelWeights> {
        &self.weights
    }
    fn kv_pos(&self) -> &StdMutex<usize> {
        &self.kv_pos
    }
    fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}

// Re-export for downstream so consumers don't need candle-core directly.
pub use candle_core::Device as CandleDevice;

// Helper for tests.
pub fn generate_from_host(
    host: &dyn session::InferenceHost,
    prompt: &str,
    max_tokens: usize,
) -> NrtResult<(String, Option<String>, u32)> {
    session::generate(host, prompt, max_tokens).map_err(Into::into)
}

// Silence dead_code for LogitsProcessor/Sampling/Tensor imports that session.rs uses.
#[allow(dead_code)]
fn _keep_imports(_: LogitsProcessor, _: Sampling, _: Tensor) {}
