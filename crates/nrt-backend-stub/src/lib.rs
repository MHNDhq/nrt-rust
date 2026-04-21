//! Deterministic in-process stub backend.
//!
//! Purpose: exercise the NRT orchestrator end-to-end without real model weights.
//! The stub simulates model load / promote / demote / infer with configurable
//! latencies so tier-transition math and co-activation timing can be measured
//! against a known ground truth.
//!
//! The router model (any model whose `ModelId` starts with `"router"` or is
//! registered via `StubBackend::register_router_intents`) returns an `intent`
//! field chosen deterministically from the prompt hash. Specialists produce
//! a canned completion keyed by their id.
//!
//! This backend is NOT a substitute for real inference in production. It is
//! instrumentation — the way `memcache`'s `MockMemcache` is to a real Memcache.

use async_trait::async_trait;
use nrt_core::{
    Backend, BackendLoadHandle, InferenceRequest, InferenceResponse, KvCacheHandle, ModelId,
    NrtError, NrtResult, Tier,
};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

/// Configurable timing profile for the stub. Defaults mirror the latency
/// targets called out in the NRT spec so benchmarks produce recognizable numbers.
#[derive(Debug, Clone)]
pub struct StubTiming {
    /// Time to load weights into Standby (system RAM). Spec implies seconds for first disk hit.
    pub load_to_standby_ms: u64,
    /// Time to promote Standby -> Resident via PCIe or unified memory.
    /// Spec: "under 50ms". Stub default: 40ms.
    pub promote_to_resident_ms: u64,
    /// Time to page a KV cache from VRAM to RAM.
    pub kv_evict_ms: u64,
    /// Time to restore a KV cache from RAM to VRAM.
    pub kv_restore_ms: u64,
    /// Per-token generation latency.
    pub per_token_ms: u64,
    /// Fixed prefill / first-token latency.
    pub prefill_ms: u64,
}

impl Default for StubTiming {
    fn default() -> Self {
        Self {
            load_to_standby_ms: 300,
            promote_to_resident_ms: 40,
            kv_evict_ms: 8,
            kv_restore_ms: 12,
            per_token_ms: 4,
            prefill_ms: 15,
        }
    }
}

impl StubTiming {
    /// Zero-latency profile for unit tests where we care about correctness, not timing.
    pub fn instant() -> Self {
        Self {
            load_to_standby_ms: 0,
            promote_to_resident_ms: 0,
            kv_evict_ms: 0,
            kv_restore_ms: 0,
            per_token_ms: 0,
            prefill_ms: 0,
        }
    }
}

#[derive(Debug)]
struct BackendState {
    timing: StubTiming,
    /// Maps router-model id -> list of intent labels it may emit.
    router_intents: HashMap<ModelId, Vec<String>>,
    load_counter: AtomicU64,
}

#[derive(Clone)]
pub struct StubBackend {
    inner: Arc<RwLock<BackendState>>,
}

impl Default for StubBackend {
    fn default() -> Self {
        Self::new(StubTiming::default())
    }
}

impl StubBackend {
    pub fn new(timing: StubTiming) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BackendState {
                timing,
                router_intents: HashMap::new(),
                load_counter: AtomicU64::new(0),
            })),
        }
    }

    /// Teach the stub which intents a given router model may emit. At inference
    /// time the stub picks one based on a stable hash of the prompt, so the
    /// same prompt always routes to the same specialist — useful for tests.
    pub fn register_router_intents(&self, router: ModelId, intents: Vec<String>) {
        self.inner.write().router_intents.insert(router, intents);
    }

    pub fn set_timing(&self, timing: StubTiming) {
        self.inner.write().timing = timing;
    }

    fn is_router(&self, id: &ModelId) -> bool {
        let state = self.inner.read();
        state.router_intents.contains_key(id) || id.as_str().starts_with("router")
    }

    fn pick_intent(&self, id: &ModelId, prompt: &str) -> Option<String> {
        let state = self.inner.read();
        let intents = state.router_intents.get(id)?;
        if intents.is_empty() {
            return None;
        }
        // Deterministic selection: djb2-like hash of the prompt mod N.
        let mut h: u32 = 5381;
        for b in prompt.as_bytes() {
            h = h.wrapping_mul(33).wrapping_add(*b as u32);
        }
        let idx = (h as usize) % intents.len();
        Some(intents[idx].clone())
    }

    fn next_token(&self) -> u64 {
        self.inner
            .read()
            .load_counter
            .fetch_add(1, Ordering::Relaxed)
    }

    fn timing(&self) -> StubTiming {
        self.inner.read().timing.clone()
    }
}

#[async_trait]
impl Backend for StubBackend {
    fn name(&self) -> &'static str {
        "stub"
    }

    async fn load(&self, model: &ModelId, tier: Tier) -> NrtResult<BackendLoadHandle> {
        let t = self.timing();
        // Stub always goes through Standby first. Callers that ask for Resident
        // pay the full (load_to_standby + promote) latency.
        tokio::time::sleep(Duration::from_millis(t.load_to_standby_ms)).await;
        let mut handle = BackendLoadHandle {
            model_id: model.clone(),
            resident_vram_mb: 0,
            standby_ram_mb: 256, // nominal 1B model at q4_k_m ~ 600 MB, pick 256 for stub
            load_token: self.next_token(),
            tier: Tier::Standby,
        };
        if matches!(tier, Tier::Resident | Tier::Active) {
            self.promote(&mut handle, Tier::Resident).await?;
        }
        if matches!(tier, Tier::Remote) {
            handle.tier = Tier::Remote;
            handle.standby_ram_mb = 0;
        }
        Ok(handle)
    }

    async fn promote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()> {
        if handle.tier == to {
            return Ok(());
        }
        match (handle.tier, to) {
            (Tier::Standby, Tier::Resident) => {
                tokio::time::sleep(Duration::from_millis(self.timing().promote_to_resident_ms))
                    .await;
                handle.resident_vram_mb = 800;
                handle.tier = Tier::Resident;
            }
            (Tier::Resident, Tier::Active) => {
                handle.tier = Tier::Active;
            }
            (Tier::Remote, Tier::Resident) => {
                return Err(NrtError::InvalidTransition {
                    from: Tier::Remote,
                    to: Tier::Resident,
                    reason: "remote model cannot be promoted locally".into(),
                });
            }
            (from, to) => {
                return Err(NrtError::InvalidTransition {
                    from,
                    to,
                    reason: "promotion path not supported".into(),
                });
            }
        }
        Ok(())
    }

    async fn demote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()> {
        if handle.tier == to {
            return Ok(());
        }
        match (handle.tier, to) {
            (Tier::Active, Tier::Resident) => {
                handle.tier = Tier::Resident;
            }
            (Tier::Resident, Tier::Standby) => {
                tokio::time::sleep(Duration::from_millis(self.timing().kv_evict_ms)).await;
                handle.resident_vram_mb = 0;
                handle.tier = Tier::Standby;
            }
            (from, to) => {
                return Err(NrtError::InvalidTransition {
                    from,
                    to,
                    reason: "demotion path not supported".into(),
                });
            }
        }
        Ok(())
    }

    async fn unload(&self, _handle: &BackendLoadHandle) -> NrtResult<()> {
        Ok(())
    }

    async fn infer(
        &self,
        handle: &BackendLoadHandle,
        _kv: Option<&KvCacheHandle>,
        req: InferenceRequest,
    ) -> NrtResult<InferenceResponse> {
        let t = self.timing();
        if !matches!(handle.tier, Tier::Resident | Tier::Active | Tier::Remote) {
            return Err(NrtError::backend(format!(
                "stub cannot infer in tier {:?}",
                handle.tier
            )));
        }

        let prefill = Duration::from_millis(t.prefill_ms);
        tokio::time::sleep(prefill).await;

        let tokens = req.max_tokens.min(64);
        let gen = Duration::from_millis(t.per_token_ms.saturating_mul(tokens as u64));
        tokio::time::sleep(gen).await;

        let intent = if self.is_router(&handle.model_id) {
            self.pick_intent(&handle.model_id, &req.prompt)
        } else {
            None
        };

        let completion = if let Some(intent) = intent.as_ref() {
            format!("intent={intent}")
        } else {
            format!(
                "[stub:{model}] ok len={plen} max={mx}",
                model = handle.model_id,
                plen = req.prompt.len(),
                mx = req.max_tokens
            )
        };

        Ok(InferenceResponse {
            session_id: req.session_id,
            model_id: req.model_id,
            completion,
            tokens_emitted: tokens,
            intent,
            latency_ms: (prefill + gen).as_millis() as u64,
        })
    }
}
