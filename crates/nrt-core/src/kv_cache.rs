use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Footprint accounting for a KV cache. The cluster's LRU policy evicts based on
/// `last_touched` and `size_bytes`; backends are free to compute size however they like.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvFootprint {
    pub size_bytes: u64,
    pub max_tokens: u32,
    pub used_tokens: u32,
}

impl KvFootprint {
    pub fn empty(max_tokens: u32) -> Self {
        Self {
            size_bytes: 0,
            max_tokens,
            used_tokens: 0,
        }
    }

    pub fn pressure(&self) -> f32 {
        if self.max_tokens == 0 {
            0.0
        } else {
            self.used_tokens as f32 / self.max_tokens as f32
        }
    }
}

/// Handle to a session's KV cache on a specific backend. The Cluster Manager
/// holds these by session id, not by raw pointer — backends own the actual tensors.
#[derive(Debug, Clone)]
pub struct KvCacheHandle {
    pub session_id: crate::SessionId,
    pub model_id: crate::ModelId,
    pub last_touched: Instant,
    pub footprint: KvFootprint,
    /// Token transferred to the backend to locate the cache on next request.
    pub backend_token: u64,
}

/// Placeholder trait for backend-owned KV caches.
/// A real CUDA backend would implement `evict_to_ram` with `cudaMemcpyAsync`.
/// Our stub backend implements these as in-memory Vec swaps.
#[async_trait::async_trait]
pub trait KvCache: Send + Sync {
    async fn allocate(
        &self,
        session: crate::SessionId,
        model: &crate::ModelId,
        max_tokens: u32,
    ) -> crate::NrtResult<KvCacheHandle>;

    async fn evict_to_ram(&self, handle: &KvCacheHandle) -> crate::NrtResult<()>;

    async fn restore_from_ram(&self, handle: &KvCacheHandle) -> crate::NrtResult<()>;

    async fn drop(&self, handle: &KvCacheHandle) -> crate::NrtResult<()>;
}
