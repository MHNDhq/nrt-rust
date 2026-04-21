use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Lightweight counters. A full deployment would export these via OpenTelemetry;
/// the prototype keeps them in-process so benchmarks can read them directly.
#[derive(Debug, Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub router_hits: AtomicU64,
    pub dispatch_hits: AtomicU64,
    pub dispatch_fallbacks: AtomicU64,
    pub promotions: AtomicU64,
    pub demotions: AtomicU64,
    pub co_activation_warms: AtomicU64,
    pub lru_evictions: AtomicU64,
    pub sessions_live: AtomicU64,
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            router_hits: self.router_hits.load(Ordering::Relaxed),
            dispatch_hits: self.dispatch_hits.load(Ordering::Relaxed),
            dispatch_fallbacks: self.dispatch_fallbacks.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            demotions: self.demotions.load(Ordering::Relaxed),
            co_activation_warms: self.co_activation_warms.load(Ordering::Relaxed),
            lru_evictions: self.lru_evictions.load(Ordering::Relaxed),
            sessions_live: self.sessions_live.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub router_hits: u64,
    pub dispatch_hits: u64,
    pub dispatch_fallbacks: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub co_activation_warms: u64,
    pub lru_evictions: u64,
    pub sessions_live: u64,
}

pub type MetricsRef = Arc<Metrics>;
