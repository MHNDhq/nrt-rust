//! LRU policy for session KV-cache eviction.
//!
//! Spec Pillar 1: "Instead of PagedAttention, NRT runs an LRU policy over
//! session KV caches. When a session goes idle, its KV cache is paged to
//! system RAM and the VRAM is reclaimed for the next active session."
//!
//! Weights stay Resident; only session state moves. This module owns the
//! policy itself; the ClusterManager owns the accounting.

use nrt_core::{ModelId, SessionId};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LruPolicy {
    /// Maximum Active sessions before we begin evicting.
    pub max_active_sessions: usize,
    /// A session idle longer than this is immediately eligible for eviction,
    /// regardless of total active count.
    pub idle_threshold: Duration,
    /// Floor on session lifetime — a newly-created session is never evicted
    /// for at least this long, even under pressure.
    pub min_lifetime: Duration,
}

impl Default for LruPolicy {
    fn default() -> Self {
        Self {
            max_active_sessions: 16,
            idle_threshold: Duration::from_secs(30),
            min_lifetime: Duration::from_millis(250),
        }
    }
}

/// Side-channel events emitted by the scheduler for benchmarking / tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerEvent {
    SessionAdmitted {
        session: SessionId,
        model: ModelId,
    },
    SessionEvicted {
        session: SessionId,
        model: ModelId,
        reason: String,
    },
    CoActivationWarmed {
        triggered_by: ModelId,
        warmed: ModelId,
    },
    Promotion {
        model: ModelId,
        from: nrt_core::Tier,
        to: nrt_core::Tier,
        elapsed_ms: u64,
    },
}
