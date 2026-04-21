//! Candle-based real SLM inference backend for NRT.
//!
//! Loads quantized GGUF weights from HuggingFace and runs real forward passes
//! through the NRT `Backend` trait. No FFI, no Python — pure Rust end-to-end.
//!
//! The backend spawns CPU work on `tokio::task::spawn_blocking` because Candle's
//! forward pass is synchronous and CPU/Metal-bound. The shared weight state is
//! protected by a `std::sync::Mutex` because the lock is held for the entire
//! forward pass.
//!
//! Each `ModelId` is backed by a `ModelProfile` describing which GGUF file to
//! fetch, which tokenizer to use, and optional intent labels the router may
//! emit. Profiles that point at the same artifacts now share one underlying
//! weight/tokenizer bundle and differ only in prompt-shaping metadata.

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
    collections::HashMap,
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
    #[error("model {0} is remote-only and has no local weights")]
    RemoteOnly(String),
    #[error("generation produced no tokens")]
    EmptyGeneration,
}

impl From<CandleError> for NrtError {
    fn from(e: CandleError) -> Self {
        NrtError::backend(e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SharedModelKey {
    gguf_repo: String,
    gguf_file: String,
    tokenizer_repo: String,
    tokenizer_file: String,
}

impl SharedModelKey {
    fn from_profile(profile: &ModelProfile) -> Self {
        Self {
            gguf_repo: profile.gguf_repo.clone(),
            gguf_file: profile.gguf_file.clone(),
            tokenizer_repo: profile.tokenizer_repo.clone(),
            tokenizer_file: profile.tokenizer_file.clone(),
        }
    }
}

struct SharedModel {
    weights: StdMutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
}

struct LoadedModel {
    shared: Arc<SharedModel>,
    shared_key: SharedModelKey,
    /// Monotonic position for kv-cache indexing on the quantized_llama model.
    kv_pos: StdMutex<usize>,
    profile: ModelProfile,
}

pub struct CandleBackend {
    device: Device,
    profiles: RwLock<ModelMap>,
    shared: RwLock<HashMap<SharedModelKey, Arc<SharedModel>>>,
    loaded: RwLock<HashMap<ModelId, Arc<LoadedModel>>>,
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
            shared: RwLock::new(Default::default()),
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
        if profile.remote_only {
            return Err(CandleError::RemoteOnly(model.as_str().to_string()));
        }

        let shared_key = SharedModelKey::from_profile(&profile);
        if let Some(shared) = self.shared.read().get(&shared_key).cloned() {
            let loaded = Arc::new(LoadedModel {
                shared,
                shared_key,
                kv_pos: StdMutex::new(0),
                profile,
            });
            self.loaded.write().insert(model.clone(), loaded.clone());
            return Ok(loaded);
        }

        let device = self.device.clone();
        let model_for_blocking = model.clone();
        let profile_for_blocking = profile.clone();

        let built_shared =
            tokio::task::spawn_blocking(move || -> Result<Arc<SharedModel>, CandleError> {
                let api = hf_hub::api::sync::Api::new().map_err(CandleError::Hub)?;
                let gguf_repo = api.repo(hf_hub::Repo::new(
                    profile_for_blocking.gguf_repo.clone(),
                    hf_hub::RepoType::Model,
                ));
                let gguf_path: PathBuf = gguf_repo
                    .get(&profile_for_blocking.gguf_file)
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

                let tokenizer =
                    Tokenizer::from_pretrained(&profile_for_blocking.tokenizer_repo, None)
                        .map_err(|e| {
                            CandleError::Tokenizer(format!(
                                "from_pretrained({:?}): {e}",
                                profile_for_blocking.tokenizer_repo
                            ))
                        })?;

                Ok(Arc::new(SharedModel {
                    weights: StdMutex::new(weights),
                    tokenizer,
                    device: device.clone(),
                }))
            })
            .await
            .map_err(|e| CandleError::Tokenizer(format!("spawn_blocking join: {e}")))??;

        let shared = {
            let mut shared_map = self.shared.write();
            shared_map
                .entry(shared_key.clone())
                .or_insert_with(|| built_shared.clone())
                .clone()
        };

        let loaded = Arc::new(LoadedModel {
            shared,
            shared_key,
            kv_pos: StdMutex::new(0),
            profile,
        });
        self.loaded.write().insert(model.clone(), loaded.clone());
        Ok(loaded)
    }

    fn next_token(&self) -> u64 {
        self.load_counter.fetch_add(1, Ordering::Relaxed)
    }

    fn warmable_model_ids(&self) -> Vec<ModelId> {
        let profiles = self.profiles.read();
        profiles
            .ids()
            .into_iter()
            .filter(|id| {
                profiles
                    .get(id)
                    .map(|profile| !profile.remote_only)
                    .unwrap_or(false)
            })
            .collect()
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
        Ok(BackendLoadHandle {
            model_id: model.clone(),
            resident_vram_mb,
            standby_ram_mb,
            load_token: self.next_token(),
            tier,
        })
    }

    async fn promote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()> {
        if handle.tier == to {
            return Ok(());
        }
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
        let removed = {
            let mut loaded = self.loaded.write();
            loaded.remove(&handle.model_id)
        };
        if let Some(loaded_model) = removed {
            let still_used = self
                .loaded
                .read()
                .values()
                .any(|candidate| candidate.shared_key == loaded_model.shared_key);
            if !still_used {
                self.shared.write().remove(&loaded_model.shared_key);
            }
        }
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
                completion: format!("[remote-fallback] would dispatch to {}", handle.model_id),
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
        let max_tokens = req.max_tokens.clamp(1, 128);

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
    /// warmable model ahead of cluster bootstrap.
    pub async fn warm_all_registered(&self) -> Result<Vec<ModelId>, CandleError> {
        let ids = self.warmable_model_ids();
        let mut warmed = Vec::with_capacity(ids.len());
        for id in &ids {
            match self.load_weights(id).await {
                Ok(_) => warmed.push(id.clone()),
                Err(e) => {
                    warn!(target: "candle", model = %id, error = %e, "prewarm failed");
                    return Err(e);
                }
            }
        }
        Ok(warmed)
    }

    /// Register a lightweight profile for Remote-tier models so `load_weights`
    /// is never called on them.
    pub fn register_remote(&self, id: ModelId) {
        let mut profile = default_tinyllama_profile();
        profile.nominal_vram_mb = 0;
        profile.nominal_ram_mb = 0;
        profile.remote_only = true;
        self.profiles.write().insert(id, profile);
    }
}

impl session::InferenceHost for LoadedModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.shared.tokenizer
    }
    fn device(&self) -> &Device {
        &self.shared.device
    }
    fn weights(&self) -> &StdMutex<ModelWeights> {
        &self.shared.weights
    }
    fn kv_pos(&self) -> &StdMutex<usize> {
        &self.kv_pos
    }
    fn profile(&self) -> &ModelProfile {
        &self.profile
    }
}

pub use candle_core::Device as CandleDevice;

pub fn generate_from_host(
    host: &dyn session::InferenceHost,
    prompt: &str,
    max_tokens: usize,
) -> NrtResult<(String, Option<String>, u32)> {
    session::generate(host, prompt, max_tokens).map_err(Into::into)
}

#[allow(dead_code)]
fn _keep_imports(_: LogitsProcessor, _: Sampling, _: Tensor) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_key_collapses_profiles_with_same_artifacts() {
        let router = router_profile(vec!["billing".into()]);
        let specialist = specialist_profile("billing");
        assert_eq!(
            SharedModelKey::from_profile(&router),
            SharedModelKey::from_profile(&specialist)
        );
    }

    #[test]
    fn warmable_model_ids_skip_remote_only_profiles() {
        let mut profiles = ModelMap::new();
        profiles.insert(
            ModelId::new("router"),
            router_profile(vec!["billing".into()]),
        );
        let backend = CandleBackend::new(profiles).expect("backend");
        backend.register_remote(ModelId::new("fallback"));

        let ids = backend.warmable_model_ids();
        assert!(ids.iter().any(|id| id.as_str() == "router"));
        assert!(!ids.iter().any(|id| id.as_str() == "fallback"));
    }
}
