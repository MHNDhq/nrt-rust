# nrt-prototype architecture

## Layering

The prototype follows the exact layering described in the NRT spec — each crate owns one layer and depends only on the layers below it:

```
+----------------------------------------------------+
| nrt-server   (axum HTTP, OpenAI-shaped subset)     |   HTTP
+----------------------------------------------------+
| nrt-cluster  (Cluster Manager, dispatch, warming)  |   Orchestration
+----------------------------------------------------+
| nrt-manifest (YAML parser, dispatch grammar)       |   Declarative config
+----------------------------------------------------+
| nrt-core     (Tier, Session, Backend, KvCache)     |   Vocabulary
+----------------------------------------------------+
| nrt-backend-stub (deterministic CPU)               |   Backend impl
+----------------------------------------------------+
```

A real deployment swaps `nrt-backend-stub` for `nrt-backend-llama-cpp`, `nrt-backend-cuda`, etc. The orchestrator never holds backend-specific types; everything goes through the `Backend` trait.

## Key data structures

`ClusterManager` is the core. It owns:

- A `DashMap<ModelId, ModelEntry>` with the live tier and load handle per model. DashMap lets the HTTP layer read the registry concurrently with the scheduler writing to it without holding shard locks across `.await`.
- A `DashMap<SessionId, Session>` with per-session state and last-touched time for LRU accounting.
- An `RwLock<Manifest>` so manifest replacement updates the runtime view instead of only reloading models.
- `Arc<dyn Backend>` — the backend that actually holds weights.
- Broadcast channel for `SchedulerEvent`s, consumed by tracing and (later) external observers.

Lock discipline: the cluster never holds a DashMap shard lock across an `.await` point. When it needs to send work to the backend, it clones the load handle out of the shard, releases the lock, performs the async call, then re-acquires the shard to write the new tier. This is the pattern that keeps throughput flat under contention.

## Dispatch

The Manifest's `dispatch_rule` is parsed into a `DispatchRule` enum at load time:

- `router.output.intent -> specialists.{intent}` becomes `IntentDispatch { source, field, target_set }`.
- `router.output.intent -> specialists.billing` becomes `FixedRoute { source, target }`.

At runtime the cluster calls `dispatch::resolve(&manifest, &router_response)` which returns `Ok(Some(target))`, `Ok(None)` (fall through to fallback), or `Err(DispatchUnresolved)` when an intent dispatch fires but the intent isn't a known specialist id.

Richer forms (conditional, fan-out, regex) would extend the enum and the parser together. The spec's "Manifest becomes a leaky abstraction" risk is mitigated here by keeping the grammar *small* and surfacing the escape hatch (the `DispatchRule::FixedRoute` variant plus application-level overrides) explicitly.

## Co-activation warming

When the router fires, the cluster spawns a detached Tokio task that walks `Manifest::all_models()` and promotes any model whose `co_activation == router` from Standby to Resident. Cost is paid off the request's critical path. The caller's first token arrives from the specialist exactly as soon as dispatch resolves. The runtime deduplicates in-flight warm tasks per model so repeated router hits do not double-promote the same Standby specialist.

Correctness is covered by the `co_activation_warms_declared_models_after_router_fires` integration test. The `warming-heavy` manifest + benchmark scenario demonstrates the firing rate under load.

## LRU sweeper

A background Tokio task runs every `LruPolicy::sweep_interval` (default 500 ms) and walks the session registry. Any session whose state is `Active`, whose lifetime is past `min_lifetime`, and whose idle time is past `idle_threshold` is marked `SessionState::Idle`, has its `KvCacheHandle` dropped, increments `lru_evictions`, and emits a `SessionEvicted` event. Weights stay Resident; only session state moves, per Pillar 1's "LRU policy over session KV caches" mechanism.

The sweeper holds only a `Weak<ClusterManager>` so it exits automatically when the last strong reference drops. On the stub backend, dropping the `KvCacheHandle` is the eviction — the stub does not allocate real KV. On a real backend (Candle + CUDA, llama-cpp), the same sweeper calls `backend.demote` to move the KV blob to system RAM.

Tests: `lru_sweeper_evicts_idle_sessions` uses a short-interval policy and asserts `lru_evictions >= 1` after an idle wait; `lru_sweeper_respects_min_lifetime` asserts a young session is protected.

## Drain-and-swap manifest replacement

`replace_manifest` now flips a `draining: AtomicBool` before the model-unload loop. While draining, `chat_completion` rejects new admissions with a retry-worthy error. The method polls `metrics.sessions_live` at 25 ms intervals until it hits zero or the configurable timeout (default 10 s) expires. Only then does the unload + reload happen. The `replace_manifest_drains_inflight_sessions` test exercises the full flow: an in-flight session blocks the swap, a new admission is refused while draining, the swap completes once the session ends.

## Budget symmetry

Bootstrap sums resident VRAM and checks against `gpu_budget`; it also sums standby RAM and checks against `ram_budget`. Both must hold or `load_models_from_manifest` returns `ClusterAtCapacity` or `RamBudgetExceeded` respectively. This is the structural sanity check; it runs once at bootstrap and is not re-checked on every admission because the manifest is immutable between `replace_manifest` calls.

## Metrics

Every interesting counter is a `std::sync::atomic::AtomicU64` under `nrt_cluster::metrics::Metrics`:

- `requests_total`, `router_hits`, `dispatch_hits`, `dispatch_fallbacks`
- `promotions`, `demotions`, `co_activation_warms`, `lru_evictions`
- `sessions_live`

Atomic counters are the minimal viable observability surface — a real deployment pipes these to OpenTelemetry. The `/v1/cluster` endpoint serializes a snapshot of tiers + metrics for quick debugging.

## What a production build would add

On top of this chassis, to reach the NRT spec's Phase 0 success criteria:

1. **Real backend** — Candle + CUDA (pure Rust) or llama.cpp FFI (`llama-cpp-2` crate) for the CPU/Metal/CUDA fast path. Replaces `StubBackend`; the orchestrator doesn't change.
2. **Kernel-level KV eviction** — the sweeper in this prototype already identifies eviction candidates; the real backend wires the eviction action to a KV-cache deallocation primitive (CUDA `cudaMemcpyAsync` to host, then `cudaFree` on device) instead of dropping a handle.
3. **Priority-aware sweep order** — `ModelRef::priority` is parsed today; the sweeper currently sorts by last-touched. A priority-aware comparator uses the `Priority` field + the `co_activation` graph to avoid evicting critical specialists under pressure.
4. **Stream-load** — `StubBackend::load` sleeps; a real backend does `mmap` + layer-by-layer scheduling. The trait signature doesn't change.
5. **Hybrid-path execution plan cache** — new crate `nrt-profiler` owns the per-(model-hash, device) DAG classification.

None of these touch `nrt-manifest` or the dispatch grammar. That's the point of Pillar 5 as the foundation — the rest of NRT plugs into a fixed orchestration surface.

## File-to-concept map

| Concept                          | Crate              | File                                   |
|----------------------------------|--------------------|----------------------------------------|
| Tier state machine               | nrt-core           | `tier.rs`                              |
| Backend trait                    | nrt-core           | `backend.rs`                           |
| KvCache trait                    | nrt-core           | `kv_cache.rs`                          |
| Manifest schema                  | nrt-manifest       | `model.rs`                             |
| Dispatch grammar                 | nrt-manifest       | `dispatch.rs`                          |
| Size parsing (`18GB`)            | nrt-manifest       | `sizes.rs`                             |
| Validation                       | nrt-manifest       | `validate.rs`                          |
| Deterministic stub backend       | nrt-backend-stub   | `lib.rs`                               |
| Cluster Manager                  | nrt-cluster        | `manager.rs`                           |
| Dispatch resolution              | nrt-cluster        | `dispatch.rs`                          |
| LRU policy shell                 | nrt-cluster        | `scheduler.rs`                         |
| Metrics                          | nrt-cluster        | `metrics.rs`                           |
| HTTP surface                     | nrt-server         | `main.rs`                              |
| Benchmark harness                | nrt-bench          | `main.rs`                              |
