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
    NrtError, NrtResult, Session, SessionId, SessionState, Tier, TierTransition,
};
use nrt_manifest::Manifest;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
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
    sessions_live: AtomicU64,
    requests_total: AtomicU64,
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
    /// Distinct live sessions currently associated with this model.
    pub sessions: u64,
    /// Successful inference calls routed to this model.
    pub requests_total: u64,
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
    manifest: RwLock<Manifest>,
    backend: Arc<dyn Backend>,
    models: DashMap<ModelId, ModelEntry>,
    sessions: DashMap<SessionId, Session>,
    pub(crate) metrics: MetricsRef,
    /// LRU policy consumed by the background sweeper to evict idle sessions.
    policy: LruPolicy,
    events_tx: broadcast::Sender<SchedulerEvent>,
    transitions: Mutex<Vec<TierTransition>>,
    warming_inflight: DashMap<ModelId, ()>,
    /// Set while replace_manifest is draining in-flight sessions. New
    /// chat_completion admissions are rejected while this is true.
    draining: AtomicBool,
}

impl ClusterManager {
    /// Build a manager from a Manifest and a Backend, loading every model declared.
    pub async fn bootstrap(
        manifest: Manifest,
        backend: Arc<dyn Backend>,
    ) -> NrtResult<ClusterHandle> {
        Self::bootstrap_with_policy(manifest, backend, LruPolicy::default()).await
    }

    /// Bootstrap variant that lets the caller override the LRU policy. Tests
    /// use this to shorten `idle_threshold` / `min_lifetime` so the sweeper
    /// fires within the test timeout.
    pub async fn bootstrap_with_policy(
        manifest: Manifest,
        backend: Arc<dyn Backend>,
        policy: LruPolicy,
    ) -> NrtResult<ClusterHandle> {
        let (tx, _rx) = broadcast::channel(256);
        let sweep_interval = policy.sweep_interval;
        let m = Arc::new(ClusterManager {
            manifest: RwLock::new(manifest.clone()),
            backend,
            models: DashMap::new(),
            sessions: DashMap::new(),
            metrics: Arc::new(Metrics::default()),
            policy,
            events_tx: tx,
            transitions: Mutex::new(Vec::new()),
            warming_inflight: DashMap::new(),
            draining: AtomicBool::new(false),
        });
        m.load_models_from_manifest(&manifest).await?;
        Self::spawn_lru_sweeper(&m, sweep_interval);
        Ok(m)
    }

    /// Replace the manifest atomically after draining in-flight sessions.
    /// Spec: "Atomic cluster updates. Pushing a new Manifest version swaps the
    /// cluster as a unit."
    ///
    /// Drain protocol: flips `draining` to true so `chat_completion` rejects
    /// new admissions, polls `sessions_live` until it hits zero (or `timeout`
    /// expires), unloads every model, rebuilds from the new manifest, then
    /// flips `draining` off.
    pub async fn replace_manifest(self: &Arc<Self>, new_manifest: Manifest) -> NrtResult<()> {
        self.replace_manifest_with_timeout(new_manifest, Duration::from_secs(10))
            .await
    }

    pub async fn replace_manifest_with_timeout(
        self: &Arc<Self>,
        new_manifest: Manifest,
        timeout: Duration,
    ) -> NrtResult<()> {
        self.draining.store(true, Ordering::Release);
        let drain_result = self.wait_for_drain(timeout).await;
        if let Err(e) = drain_result {
            self.draining.store(false, Ordering::Release);
            return Err(e);
        }

        for kv in self.models.iter() {
            self.backend.unload(&kv.value().handle).await?;
        }
        self.models.clear();
        self.transitions.lock().clear();
        self.warming_inflight.clear();
        let load_result = self.load_models_from_manifest(&new_manifest).await;
        if let Err(e) = load_result {
            self.draining.store(false, Ordering::Release);
            return Err(e);
        }
        *self.manifest.write() = new_manifest;
        self.draining.store(false, Ordering::Release);
        Ok(())
    }

    async fn wait_for_drain(&self, timeout: Duration) -> NrtResult<()> {
        let start = Instant::now();
        loop {
            let live = self.metrics.sessions_live.load(Ordering::Relaxed);
            if live == 0 {
                return Ok(());
            }
            if start.elapsed() >= timeout {
                return Err(NrtError::backend(format!(
                    "manifest swap drain timed out: {live} sessions still live after {} ms",
                    timeout.as_millis()
                )));
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<SchedulerEvent> {
        self.events_tx.subscribe()
    }

    pub fn snapshot(&self) -> ClusterSnapshot {
        let manifest = self.manifest();
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
                    sessions: kv.value().sessions_live.load(Ordering::Relaxed),
                    requests_total: kv.value().requests_total.load(Ordering::Relaxed),
                }
            })
            .collect();
        ClusterSnapshot {
            cluster: manifest.cluster,
            models,
            sessions_live: self.sessions.len(),
            metrics: self.metrics.snapshot(),
        }
    }

    pub fn manifest(&self) -> Manifest {
        self.manifest.read().clone()
    }

    pub fn transitions(&self) -> Vec<TierTransition> {
        self.transitions.lock().clone()
    }

    async fn load_models_from_manifest(self: &Arc<Self>, manifest: &Manifest) -> NrtResult<()> {
        let total_resident_mb: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let total_standby_mb: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        for (i, model_ref) in manifest.all_models().into_iter().enumerate() {
            let target_tier = if model_ref.tier == Tier::Active {
                // Active is a session concept, not a residency concept. Coerce to Resident.
                Tier::Resident
            } else {
                model_ref.tier
            };
            let start = Instant::now();
            let handle = self.backend.load(&model_ref.id, target_tier).await?;
            let elapsed_ms = start.elapsed().as_millis() as u64;
            self.metrics.promotions.fetch_add(1, Ordering::Relaxed);
            total_resident_mb.fetch_add(handle.resident_vram_mb, Ordering::Relaxed);
            total_standby_mb.fetch_add(handle.standby_ram_mb, Ordering::Relaxed);
            self.record_transition(
                &model_ref.id,
                Tier::Standby,
                handle.tier,
                elapsed_ms,
                "bootstrap",
            );
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
                    sessions_live: AtomicU64::new(0),
                    requests_total: AtomicU64::new(0),
                },
            );
        }

        let resident_mb = total_resident_mb.load(Ordering::Relaxed);
        let standby_mb = total_standby_mb.load(Ordering::Relaxed);
        let budget_mb = manifest.resources.gpu_budget_mb;
        let ram_budget_mb = manifest.resources.ram_budget_mb;
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
        if standby_mb > ram_budget_mb {
            return Err(NrtError::RamBudgetExceeded {
                standby_mb,
                budget_ram_mb: ram_budget_mb,
            });
        }

        info!(
            target: "cluster",
            cluster = %manifest.cluster,
            models = self.models.len(),
            resident_mb,
            standby_mb,
            budget_mb,
            ram_budget_mb,
            "cluster bootstrap complete"
        );
        Ok(())
    }

    fn record_transition(
        &self,
        model: &ModelId,
        from: Tier,
        to: Tier,
        elapsed_ms: u64,
        reason: &str,
    ) {
        self.transitions.lock().push(TierTransition {
            model_id: model.clone(),
            from,
            to,
            elapsed_ms,
            reason: reason.to_string(),
        });
    }

    /// Fire the main chat-completions flow: entry model -> dispatch -> specialist.
    #[instrument(skip(self, prompt))]
    pub async fn chat_completion(
        self: &Arc<Self>,
        session_id: Option<SessionId>,
        prompt: String,
        max_tokens: u32,
    ) -> NrtResult<CompletionResult> {
        if self.draining.load(Ordering::Acquire) {
            return Err(NrtError::backend(
                "cluster is draining for a manifest swap; retry shortly",
            ));
        }
        let ephemeral_session = session_id.is_none();
        let session_id = session_id.unwrap_or_default();
        let manifest = self.manifest();
        let entry = manifest.routing.entry.clone();
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();

        self.register_session_if_new(session_id, entry.clone());

        let result = async {
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
            self.spawn_co_activation_warm(manifest.clone(), &entry);

            let target_id = match dispatch::resolve(&manifest, &router_resp) {
                Ok(Some(t)) => t,
                Ok(None) | Err(NrtError::DispatchUnresolved { .. }) => {
                    self.metrics
                        .dispatch_fallbacks
                        .fetch_add(1, Ordering::Relaxed);
                    if let Some(fb) = manifest.fallback.as_ref() {
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
        .await;

        if ephemeral_session {
            let _ = self.end_session(session_id).await;
        }

        result
    }

    fn register_session_if_new(&self, id: SessionId, model: ModelId) {
        match self.sessions.entry(id) {
            dashmap::mapref::entry::Entry::Occupied(_) => {}
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(Session::new(id, model.clone()));
                self.metrics.sessions_live.fetch_add(1, Ordering::Relaxed);
                if let Some(model_entry) = self.models.get(&model) {
                    model_entry
                        .value()
                        .sessions_live
                        .fetch_add(1, Ordering::Relaxed);
                }
                let _ = self
                    .events_tx
                    .send(SchedulerEvent::SessionAdmitted { session: id, model });
            }
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
            let entry = self
                .models
                .get(model_id)
                .ok_or_else(|| NrtError::ModelNotFound(model_id.as_str().to_string()))?;
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

        // For v0, KV cache plumbing is not wired through yet. The prototype
        // passes no backend KV handle, so every request runs as a fresh prompt
        // execution even when the caller reuses a session id.
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
        let model_was_new = if let Some(mut session) = self.sessions.get_mut(&session_id) {
            session.value_mut().touch_model(model_id.clone())
        } else {
            false
        };
        if let Some(entry) = self.models.get(model_id) {
            let entry_ref = entry.value();
            if model_was_new {
                entry_ref.sessions_live.fetch_add(1, Ordering::Relaxed);
            }
            entry_ref.requests_total.fetch_add(1, Ordering::Relaxed);
        }

        Ok(resp)
    }

    /// Fire-and-forget: for each model in the manifest whose `co_activation`
    /// field names `trigger`, promote it to Resident (if currently Standby)
    /// without blocking the caller.
    fn spawn_co_activation_warm(self: &Arc<Self>, manifest: Manifest, trigger: &ModelId) {
        let me = self.clone();
        let trigger = trigger.clone();
        tokio::spawn(async move {
            let to_warm: Vec<ModelId> =
                me.models_to_warm(&manifest, &trigger).into_iter().collect();
            for id in to_warm {
                if let Err(err) = me.warm_if_standby(&trigger, &id).await {
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

    fn models_to_warm(&self, manifest: &Manifest, trigger: &ModelId) -> Vec<ModelId> {
        manifest
            .all_models()
            .iter()
            .filter(|m| m.co_activation.as_ref() == Some(trigger))
            .map(|m| m.id.clone())
            .collect()
    }

    async fn warm_if_standby(
        self: &Arc<Self>,
        trigger: &ModelId,
        model_id: &ModelId,
    ) -> NrtResult<()> {
        if self.warming_inflight.insert(model_id.clone(), ()).is_some() {
            return Ok(());
        }

        let result = async {
            let mut handle = {
                let entry = self
                    .models
                    .get(model_id)
                    .ok_or_else(|| NrtError::ModelNotFound(model_id.as_str().to_string()))?;
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
                triggered_by: trigger.clone(),
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
        .await;

        self.warming_inflight.remove(model_id);
        result
    }

    /// End a session — drop its KV cache and decrement live count.
    pub async fn end_session(&self, session_id: SessionId) -> NrtResult<()> {
        if let Some((_id, session)) = self.sessions.remove(&session_id) {
            self.metrics.sessions_live.fetch_sub(1, Ordering::Relaxed);
            for model_id in session.models_touched {
                if let Some(model_entry) = self.models.get(&model_id) {
                    model_entry
                        .value()
                        .sessions_live
                        .fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    /// Background LRU sweeper. Walks live sessions and evicts the KV cache of
    /// any session idle past `policy.idle_threshold` (subject to
    /// `policy.min_lifetime`). The sweeper does not unload weights — weights
    /// remain Resident per Pillar 1, only session state moves to Evicted.
    ///
    /// The task holds a `Weak` handle so it exits automatically when the last
    /// `Arc<ClusterManager>` is dropped.
    fn spawn_lru_sweeper(m: &ClusterHandle, interval: Duration) {
        let weak = Arc::downgrade(m);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(manager) = weak.upgrade() else {
                    break;
                };
                manager.sweep_lru_once();
            }
        });
    }

    fn sweep_lru_once(&self) {
        let now = Instant::now();
        let idle_threshold = self.policy.idle_threshold;
        let min_lifetime = self.policy.min_lifetime;

        let candidates: Vec<SessionId> = self
            .sessions
            .iter()
            .filter_map(|kv| {
                let s = kv.value();
                if s.state != SessionState::Active {
                    return None;
                }
                let lived = now.duration_since(s.created_at);
                let idle = now.duration_since(s.last_request_at);
                if lived >= min_lifetime && idle >= idle_threshold {
                    Some(s.id)
                } else {
                    None
                }
            })
            .collect();

        if candidates.is_empty() {
            return;
        }

        let mut evicted = 0u64;
        let mut events = Vec::new();
        for id in candidates {
            if let Some(mut entry) = self.sessions.get_mut(&id) {
                let s = entry.value_mut();
                if s.state == SessionState::Active {
                    s.state = SessionState::Idle;
                    s.kv = None;
                    let model = s.model_id.clone();
                    evicted += 1;
                    events.push((id, model));
                }
            }
        }

        if evicted > 0 {
            self.metrics
                .lru_evictions
                .fetch_add(evicted, Ordering::Relaxed);
            for (session, model) in events {
                let _ = self.events_tx.send(SchedulerEvent::SessionEvicted {
                    session,
                    model,
                    reason: "idle".into(),
                });
            }
            debug!(target: "cluster", evicted, "lru sweep evicted sessions");
        }
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
            let entry = self
                .models
                .get(model_id)
                .ok_or_else(|| NrtError::ModelNotFound(model_id.as_str().to_string()))?;
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
