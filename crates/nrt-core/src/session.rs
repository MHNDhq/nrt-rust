use crate::{KvCacheHandle, ModelId, SessionId};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, time::Instant};

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
    pub models_touched: HashSet<ModelId>,
    pub state: SessionState,
    pub kv: Option<KvCacheHandle>,
    pub created_at: Instant,
    pub last_request_at: Instant,
    pub request_count: u64,
}

impl Session {
    pub fn new(id: SessionId, model_id: ModelId) -> Self {
        let now = Instant::now();
        let mut models_touched = HashSet::new();
        models_touched.insert(model_id.clone());
        Self {
            id,
            model_id,
            models_touched,
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

    /// Record that this session executed work on `model`. Returns true when the
    /// model is newly associated with the live session.
    pub fn touch_model(&mut self, model: ModelId) -> bool {
        self.model_id = model.clone();
        let inserted = self.models_touched.insert(model);
        self.touch();
        inserted
    }

    pub fn idle_ms(&self) -> u64 {
        self.last_request_at.elapsed().as_millis() as u64
    }
}
