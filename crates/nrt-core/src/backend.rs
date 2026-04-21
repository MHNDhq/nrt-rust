use crate::{KvCacheHandle, ModelId, NrtResult, SessionId, Tier};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub session_id: SessionId,
    pub model_id: ModelId,
    pub prompt: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Extra fields the backend may use: tool schema, stop sequences, etc.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub session_id: SessionId,
    pub model_id: ModelId,
    pub completion: String,
    pub tokens_emitted: u32,
    /// Backend-declared intent label (for router models). The Cluster's dispatch
    /// rule maps this to a specialist. Non-router models return None.
    #[serde(default)]
    pub intent: Option<String>,
    pub latency_ms: u64,
}

/// A single streamed token (for future streaming endpoint).
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub log_prob: f32,
    pub is_final: bool,
}

/// Opaque handle returned after loading weights. Backends are free to make this
/// internally ref-counted; the Cluster Manager holds it for tier accounting.
#[derive(Debug, Clone)]
pub struct BackendLoadHandle {
    pub model_id: ModelId,
    pub resident_vram_mb: u64,
    pub standby_ram_mb: u64,
    pub load_token: u64,
    pub tier: Tier,
}

/// Minimal surface every backend must implement. Real backends (llama.cpp FFI,
/// CUDA, Metal, ANE) bolt into this trait. The stub backend in this prototype
/// implements it with deterministic in-process behavior so the rest of the
/// system is exercisable end-to-end without real model weights.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Name used in logs and the HTTP info endpoint ("stub", "llama-cpp", "cuda", ...).
    fn name(&self) -> &'static str;

    /// Load weights into the requested tier. A backend is free to reject a tier
    /// it doesn't support (e.g., a CPU-only backend rejecting Resident VRAM tier).
    async fn load(&self, model: &ModelId, tier: Tier) -> NrtResult<BackendLoadHandle>;

    /// Promote from Standby to Resident (or Resident to Active). The Cluster
    /// calls this on co-activation warming before it has a request in hand.
    async fn promote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()>;

    /// Demote in the reverse direction. Typical when VRAM pressure exceeds budget.
    async fn demote(&self, handle: &mut BackendLoadHandle, to: Tier) -> NrtResult<()>;

    /// Unload completely. Called on cluster shutdown or explicit removal.
    async fn unload(&self, handle: &BackendLoadHandle) -> NrtResult<()>;

    /// Single-shot inference. Streaming is a future milestone.
    async fn infer(
        &self,
        handle: &BackendLoadHandle,
        kv: Option<&KvCacheHandle>,
        req: InferenceRequest,
    ) -> NrtResult<InferenceResponse>;
}
