use crate::{KvCacheHandle, ModelId, SessionId};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    /// KV cache is live on a backend.
    Active,
    /// KV cache was paged to system RAM; weights still Resident but no generation in flight.
    Idle,
    /// Evicted entirely; restoring means re-running the prompt prefix.
    Dropped,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub model_id: ModelId,
    pub state: SessionState,
    pub kv: Option<KvCacheHandle>,
    pub created_at: Instant,
    pub last_request_at: Instant,
    pub request_count: u64,
}

impl Session {
    pub fn new(id: SessionId, model_id: ModelId) -> Self {
        let now = Instant::now();
        Self {
            id,
            model_id,
            state: SessionState::Active,
            kv: None,
            created_at: now,
            last_request_at: now,
            request_count: 0,
        }
    }

    pub fn touch(&mut self) {
        self.last_request_at = Instant::now();
        self.request_count += 1;
    }

    pub fn idle_ms(&self) -> u64 {
        self.last_request_at.elapsed().as_millis() as u64
    }
}
