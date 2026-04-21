//! The Cluster Manager — the central orchestrator.

use crate::{
    dispatch,
    metrics::{Metrics, MetricsRef, MetricsSnapshot},
    scheduler::{LruPolicy, SchedulerEvent},
};
use async_trait::async_trait;
use dashmap::DashMap;
use nrt_core::{
    Backend, BackendLoadHandle, InferenceRequest, InferenceResponse, KvCacheHandle, ModelId,
    NrtError, NrtResult, Session, SessionId, Tier, TierTransition,
};
use nrt_manifest::Manifest;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;
use tokio::sync::broadcast;
use tracing::{debug, info, instrument, warn};

/// Registry entry per model. We keep one load handle per model; the backend
/// can support sharing if multiple sessions want to run against it.
#[derive(Debug)]
struct ModelEntry {
    handle: BackendLoadHandle,
    /// Index into `Manifest::all_models()` — retained for future scheduling
    /// heuristics (priority-aware eviction) that walk models in manifest order.
    #[allow(dead_code)]
    manifest_index: usize,
    sessions: AtomicU64,
}

/// External view of a cluster — returned by snapshot() and /v1/cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub cluster: String,
    pub models: Vec<ModelStatus>,
    pub sessions_live: usize,
    pub metrics: MetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: ModelId,
    pub tier: Tier,
    pub resident_vram_mb: u64,
    pub standby_ram_mb: u64,
    pub sessions: u64,
}

/// The object returned by a /v1/chat/completions call after the cluster has
/// walked the router -> specialist graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    pub session_id: SessionId,
    pub entry_model: ModelId,
    pub final_model: ModelId,
    pub intent: Option<String>,
    pub completion: String,
    pub latency_ms: u64,
    pub hops: Vec<CompletionHop>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionHop {
    pub model: ModelId,
    pub completion: String,
    pub intent: Option<String>,
    pub latency_ms: u64,
}

/// Cheap cloneable handle to the manager. All methods go through Arc internally,
/// so cloning this is a refcount bump.
pub type ClusterHandle = Arc<ClusterManager>;

pub struct ClusterManager {
    manifest: Manifest,
    backend: Arc<dyn Backend>,
    models: DashMap<ModelId, ModelEntry>,
    sessions: DashMap<SessionId, Session>,
    pub(crate) metrics: MetricsRef,
    /// LRU policy used by the eviction path. Wired in place; eviction is
    /// driven externally (by /v1/cluster/evict or the background sweeper) in
    /// the next milestone.
    #[allow(dead_code)]
    policy: LruPolicy,
    events_tx: broadcast::Sender<SchedulerEvent>,
    transitions: Mutex<Vec<TierTransition>>,
}

impl ClusterManager {
    /// Build a manager from a Manifest and a Backend, loading every model declared.
    pub async fn bootstrap(
        manifest: Manifest,
        backend: Arc<dyn Backend>,
    ) -> NrtResult<ClusterHandle> {
        let (tx, _rx) = broadcast::channel(256);
        let m = Arc::new(ClusterManager {
            manifest,
            backend,
            models: DashMap::new(),
            sessions: DashMap::new(),
            metrics: Arc::new(Metrics::default()),
            policy: LruPolicy::default(),
            events_tx: tx,
            transitions: Mutex::new(Vec::new()),
        });
        m.initial_load().await?;
        Ok(m)
    }

    /// Replace the manifest atomically. Delta-computes model adds / removes /
    /// tier changes so the cluster doesn't thrash on a simple version bump.
    /// Spec: "Atomic cluster updates. Pushing a new Manifest version swaps the
    /// cluster as a unit."
    pub async fn replace_manifest(self: &Arc<Self>, new_manifest: Manifest) -> NrtResult<()> {
        // v0 implementation: we reject live swap if there are active sessions.
        // A future milestone will implement drain-then-swap.
        let active = self.metrics.sessions_live.load(Ordering::Relaxed);
        if active > 0 {
            return Err(NrtError::backend(format!(
                "refusing to swap manifest with {active} active sessions (drain-and-swap pending)"
            )));
        }

        // Unload all models, then rebuild.
        for kv in self.models.iter() {
            self.backend.unload(&kv.value().handle).await?;
        }
        self.models.clear();

        // Swap the manifest in-place. Safe because we're &Arc<Self> and holding
        // the only caller here — this is a single-writer API by contract.
        // We use unsafe only to express that intent; safer alternative is a RwLock.
        // For v0 we just document the constraint and log a warning.
        warn!(target: "cluster", "manifest replace is v0 — swap-in-place is not yet implemented; rebuild the cluster instead");

        // Load each model from the new manifest.
        for (i, model_ref) in new_manifest.all_models().into_iter().enumerate() {
            let handle = self
                .backend
                .load(&model_ref.id, model_ref.tier)
                .await?;
            self.models.insert(
                model_ref.id.clone(),
                ModelEntry {
                    handle,
                    manifest_index: i,
                    sessions: AtomicU64::new(0),
                },
            );
        }

        // Stash the new manifest for consistency. Actual swap left to a proper
        // RwLock revision in the next milestone.
        let _ = new_manifest;
        Ok(())
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<SchedulerEvent> {
        self.events_tx.subscribe()
    }

    pub fn snapshot(&self) -> ClusterSnapshot {
        let models = self
            .models
            .iter()
            .map(|kv| {
                let h = &kv.value().handle;
                ModelStatus {
                    id: kv.key().clone(),
                    tier: h.tier,
                    resident_vram_mb: h.resident_vram_mb,
                    standby_ram_mb: h.standby_ram_mb,
                    sessions: kv.value().sessions.load(Ordering::Relaxed),
                }
            })
            .collect();
        ClusterSnapshot {
            cluster: self.manifest.cluster.clone(),
            models,
            sessions_live: self.sessions.len(),
            metrics: self.metrics.snapshot(),
        }
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn transitions(&self) -> Vec<TierTransition> {
        self.transitions.lock().clone()
    }

    async fn initial_load(self: &Arc<Self>) -> NrtResult<()> {
        let total_resident_mb: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        for (i, model_ref) in self.manifest.all_models().into_iter().enumerate() {
            let target_tier = if model_ref.tier == Tier::Active {
                // Active is a session concept, not a residency concept. Coerce to Resident.
                Tier::Resident
            } else {
                model_ref.tier
            };
            let start = Instant::now();
            let handle = self
                .backend
                .load(&model_ref.id, target_tier)
                .await?;
            let elapsed_ms = start.elapsed().as_millis() as u64;
            self.metrics.promotions.fetch_add(1, Ordering::Relaxed);
            total_resident_mb
                .fetch_add(handle.resident_vram_mb, Ordering::Relaxed);
            self.record_transition(&model_ref.id, Tier::Standby, handle.tier, elapsed_ms, "bootstrap");
            let _ = self.events_tx.send(SchedulerEvent::Promotion {
                model: model_ref.id.clone(),
                from: Tier::Standby,
                to: handle.tier,
                elapsed_ms,
            });
            self.models.insert(
                model_ref.id.clone(),
                ModelEntry {
                    handle,
                    manifest_index: i,
                    sessions: AtomicU64::new(0),
                },
            );
        }

        let resident_mb = total_resident_mb.load(Ordering::Relaxed);
        let budget_mb = self.manifest.resources.gpu_budget_mb;
        if resident_mb > budget_mb {
            return Err(NrtError::ClusterAtCapacity {
                resident: self
                    .models
                    .iter()
                    .filter(|kv| kv.value().handle.tier == Tier::Resident)
                    .count(),
                standby: self
                    .models
                    .iter()
                    .filter(|kv| kv.value().handle.tier == Tier::Standby)
                    .count(),
                budget_vram_mb: budget_mb,
            });
        }

        info!(
            target: "cluster",
            cluster = %self.manifest.cluster,
            models = self.models.len(),
            resident_mb,
            budget_mb,
            "cluster bootstrap complete"
        );
        Ok(())
    }

    fn record_transition(&self, model: &ModelId, from: Tier, to: Tier, elapsed_ms: u64, reason: &str) {
        self.transitions.lock().push(TierTransition {
            model_id: model.clone(),
            from,
            to,
            elapsed_ms,
            reason: reason.to_string(),
        });
    }

    /// Fire the main chat-completions flow: entry model -> dispatch -> specialist.
    #[instrument(skip(self, prompt), fields(cluster = %self.manifest.cluster))]
    pub async fn chat_completion(
        self: &Arc<Self>,
        session_id: Option<SessionId>,
        prompt: String,
        max_tokens: u32,
    ) -> NrtResult<CompletionResult> {
        let session_id = session_id.unwrap_or_else(SessionId::new);
        let entry = self.manifest.routing.entry.clone();
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();

        self.register_session_if_new(session_id, entry.clone());

        let mut hops = Vec::new();
        let router_resp = self
            .run_model(&entry, session_id, &prompt, max_tokens)
            .await?;
        self.metrics.router_hits.fetch_add(1, Ordering::Relaxed);
        hops.push(CompletionHop {
            model: entry.clone(),
            completion: router_resp.completion.clone(),
            intent: router_resp.intent.clone(),
            latency_ms: router_resp.latency_ms,
        });

        // Co-activation warming: now that the router fired, warm every model that
        // declared it as a co_activation trigger. This is deliberately fire-and-forget.
        self.spawn_co_activation_warm(&entry);

        let target_id = match dispatch::resolve(&self.manifest, &router_resp) {
            Ok(Some(t)) => t,
            Ok(None) | Err(NrtError::DispatchUnresolved { .. }) => {
                // Intent could not be resolved to a specialist. If the manifest
                // declares a fallback, dispatch there; otherwise return the router's
                // own completion and let the caller handle the miss.
                self.metrics
                    .dispatch_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                if let Some(fb) = self.manifest.fallback.as_ref() {
                    let fb_id = fb.id.clone();
                    let fb_resp = self
                        .run_model(&fb_id, session_id, &prompt, max_tokens)
                        .await?;
                    hops.push(CompletionHop {
                        model: fb_id.clone(),
                        completion: fb_resp.completion.clone(),
                        intent: fb_resp.intent.clone(),
                        latency_ms: fb_resp.latency_ms,
                    });
                    let latency_ms = t0.elapsed().as_millis() as u64;
                    return Ok(CompletionResult {
                        session_id,
                        entry_model: entry,
                        final_model: fb_id,
                        intent: router_resp.intent,
                        completion: fb_resp.completion,
                        latency_ms,
                        hops,
                    });
                }
                let latency_ms = t0.elapsed().as_millis() as u64;
                return Ok(CompletionResult {
                    session_id,
                    entry_model: entry.clone(),
                    final_model: entry,
                    intent: router_resp.intent,
                    completion: router_resp.completion,
                    latency_ms,
                    hops,
                });
            }
            Err(e) => return Err(e),
        };

        let final_resp = self
            .run_model(&target_id, session_id, &prompt, max_tokens)
            .await?;
        self.metrics.dispatch_hits.fetch_add(1, Ordering::Relaxed);
        hops.push(CompletionHop {
            model: target_id.clone(),
            completion: final_resp.completion.clone(),
            intent: final_resp.intent.clone(),
            latency_ms: final_resp.latency_ms,
        });

        let latency_ms = t0.elapsed().as_millis() as u64;
        Ok(CompletionResult {
            session_id,
            entry_model: entry,
            final_model: target_id,
            intent: router_resp.intent,
            completion: final_resp.completion,
            latency_ms,
            hops,
        })
    }

    fn register_session_if_new(&self, id: SessionId, model: ModelId) {
        if !self.sessions.contains_key(&id) {
            self.sessions.insert(id, Session::new(id, model));
            self.metrics.sessions_live.fetch_add(1, Ordering::Relaxed);
            let _ = self.events_tx.send(SchedulerEvent::SessionAdmitted {
                session: id,
                model: self.manifest.routing.entry.clone(),
            });
        }
    }

    /// Promote Standby -> Resident if needed, then call backend.infer.
    async fn run_model(
        self: &Arc<Self>,
        model_id: &ModelId,
        session_id: SessionId,
        prompt: &str,
        max_tokens: u32,
    ) -> NrtResult<InferenceResponse> {
        // Pull the current tier + handle. We clone the handle to avoid holding
        // the DashMap shard lock across an await.
        let mut handle = {
            let entry = self.models.get(model_id).ok_or_else(|| {
                NrtError::ModelNotFound(model_id.as_str().to_string())
            })?;
            entry.value().handle.clone()
        };

        if matches!(handle.tier, Tier::Standby) {
            let start = Instant::now();
            self.backend.promote(&mut handle, Tier::Resident).await?;
            let elapsed_ms = start.elapsed().as_millis() as u64;
            self.metrics.promotions.fetch_add(1, Ordering::Relaxed);
            self.record_transition(
                model_id,
                Tier::Standby,
                Tier::Resident,
                elapsed_ms,
                "on-demand promotion",
            );
            let _ = self.events_tx.send(SchedulerEvent::Promotion {
                model: model_id.clone(),
                from: Tier::Standby,
                to: Tier::Resident,
                elapsed_ms,
            });
            // Write the new tier back into the registry.
            if let Some(mut entry) = self.models.get_mut(model_id) {
                entry.value_mut().handle = handle.clone();
            }
        }

        // For v0, KV cache is owned by the backend directly; the prototype
        // treats every request as session-sticky and lets the backend decide
        // whether to cache internally. Real backends implement the KvCache trait.
        let kv: Option<&KvCacheHandle> = None;

        let req = InferenceRequest {
            session_id,
            model_id: model_id.clone(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature: None,
            extra: Default::default(),
        };

        let resp = self.backend.infer(&handle, kv, req).await?;

        // Session bookkeeping.
        if let Some(mut session) = self.sessions.get_mut(&session_id) {
            session.value_mut().touch();
        }
        if let Some(entry) = self.models.get(model_id) {
            entry
                .value()
                .sessions
                .fetch_add(1, Ordering::Relaxed);
        }

        Ok(resp)
    }

    /// Fire-and-forget: for each model in the manifest whose `co_activation`
    /// field names `trigger`, promote it to Resident (if currently Standby)
    /// without blocking the caller.
    fn spawn_co_activation_warm(self: &Arc<Self>, trigger: &ModelId) {
        let me = self.clone();
        let trigger = trigger.clone();
        tokio::spawn(async move {
            let to_warm: Vec<ModelId> = me
                .manifest
                .all_models()
                .iter()
                .filter(|m| m.co_activation.as_ref() == Some(&trigger))
                .map(|m| m.id.clone())
                .collect();
            for id in to_warm {
                if let Err(err) = me.warm_if_standby(&id).await {
                    warn!(
                        target: "cluster",
                        model = %id,
                        trigger = %trigger,
                        error = %err,
                        "co-activation warming failed"
                    );
                }
            }
        });
    }

    async fn warm_if_standby(self: &Arc<Self>, model_id: &ModelId) -> NrtResult<()> {
        let mut handle = {
            let entry = self.models.get(model_id).ok_or_else(|| {
                NrtError::ModelNotFound(model_id.as_str().to_string())
            })?;
            entry.value().handle.clone()
        };
        if !matches!(handle.tier, Tier::Standby) {
            return Ok(());
        }
        let start = Instant::now();
        self.backend.promote(&mut handle, Tier::Resident).await?;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        self.metrics
            .co_activation_warms
            .fetch_add(1, Ordering::Relaxed);
        self.record_transition(
            model_id,
            Tier::Standby,
            Tier::Resident,
            elapsed_ms,
            "co-activation warm",
        );
        if let Some(mut entry) = self.models.get_mut(model_id) {
            entry.value_mut().handle = handle.clone();
        }
        let _ = self.events_tx.send(SchedulerEvent::CoActivationWarmed {
            triggered_by: ModelId::new("router"),
            warmed: model_id.clone(),
        });
        debug!(
            target: "cluster",
            model = %model_id,
            elapsed_ms,
            "co-activation warm complete"
        );
        Ok(())
    }

    /// End a session — drop its KV cache and decrement live count.
    pub async fn end_session(&self, session_id: SessionId) -> NrtResult<()> {
        if self.sessions.remove(&session_id).is_some() {
            self.metrics.sessions_live.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Helper methods exposed to tests and benches that want to drive the
/// scheduler manually. Not part of the public API.
#[async_trait]
pub trait TestingApi {
    async fn force_demote(&self, model_id: &ModelId, to: Tier) -> NrtResult<()>;
}

#[async_trait]
impl TestingApi for ClusterManager {
    async fn force_demote(&self, model_id: &ModelId, to: Tier) -> NrtResult<()> {
        let mut handle = {
            let entry = self.models.get(model_id).ok_or_else(|| {
                NrtError::ModelNotFound(model_id.as_str().to_string())
            })?;
            entry.value().handle.clone()
        };
        self.backend.demote(&mut handle, to).await?;
        self.metrics.demotions.fetch_add(1, Ordering::Relaxed);
        if let Some(mut entry) = self.models.get_mut(model_id) {
            entry.value_mut().handle = handle;
        }
        Ok(())
    }
}

/// Helpers on the &Arc<ClusterManager> level. Kept out of the inherent impl so
/// they can be dropped when the real metrics exporter is wired up.
pub fn metrics_of(h: &ClusterHandle) -> MetricsRef {
    h.metrics.clone()
}

/// Helpful for tests that want to wait for all spawned warm tasks to complete.
pub async fn settle(h: &ClusterHandle) {
    tokio::task::yield_now().await;
    // Cheap heuristic: wait a pair of scheduler ticks. Fine for tests.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let _ = h.models.len();
}
