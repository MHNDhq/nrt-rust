# nrt-prototype

A Rust prototype of the **Neurometric Runtime (NRT)** Pillar 5 — the Manifest-driven Cluster Manager — built against the NRT product spec v2, with real SLM inference via a pure-Rust Candle backend and a live cluster dashboard.

This is a proof-of-build artifact that accompanies a proposal for the **Rust Agentic Inference Engine for SLMs** Upwork job. It is not a full Phase 0 — see [Scope](#scope).

## What it does

- Parses the exact `nrt.yaml` example from the NRT spec (router + specialists + fallback + dispatch rule).
- Boots a cluster of stub models and places each at its declared tier (`resident` / `standby` / `remote`).
- Exposes an **OpenAI-compatible `/v1/chat/completions`** endpoint backed by the cluster.
- Fires the router, resolves the dispatch rule (`router.output.intent -> specialists.{intent}`), then invokes the resolved specialist in a single HTTP round-trip.
- Performs **Manifest-declared co-activation warming** (fire-and-forget Standby → Resident promotion for any specialist whose `co_activation` field names the just-fired router).
- Records tier transitions, promotions, co-activation warms, dispatch hits, and session lifetimes as structured metrics.

Also included:

- **Real SLM inference** via `nrt-backend-candle`: pure-Rust inference over TinyLlama-1.1B Q4_K_M weights loaded from HuggingFace. No Python, no FFI to llama.cpp. `cargo run -p nrt-backend-candle --example hello_candle` downloads the weights (first run, ~670 MB) and generates real tokens in a forward pass.
- **Live cluster dashboard** served at `http://127.0.0.1:9000/` (when the server runs). Dark, IBM Plex Sans / Plex Mono, shows live tier state, VRAM gauges, metrics, and a request log with the router-to-specialist hop breakdown.

It does **not** do (these are follow-on milestones — see [Scope](#scope)):

- CUDA / Metal / ANE / Hexagon kernels beyond what Candle exposes by default. A Candle Metal backend is a cargo-feature flip away but was not enabled in this prototype to keep the demo CPU-portable.
- Stream-load with memory-mapped weights, JIT quantization, or background upgrade.
- PagedAttention (Pillar 4 linear allocator is sketched as a trait shape; no real kernel).

## Workspace layout

```
nrt-prototype/
├── Cargo.toml                      # workspace
├── rust-toolchain.toml             # stable
├── crates/
│   ├── nrt-core/                   # Tier, SessionId, Backend trait, KvCache trait
│   ├── nrt-manifest/               # nrt.yaml parser + validator + dispatch grammar
│   ├── nrt-backend-stub/           # deterministic CPU stub backend (for benches)
│   ├── nrt-backend-candle/         # real SLM inference via HuggingFace Candle
│   ├── nrt-cluster/                # Cluster Manager — the brain
│   ├── nrt-server/                 # axum HTTP + embedded live dashboard
│   └── nrt-bench/                  # benchmark harness
├── manifests/
│   ├── customer-support.yaml       # verbatim from the spec
│   ├── density-50.yaml             # 51 models at 20.8 GB VRAM nominal
│   └── warming-heavy.yaml          # 8 standby specialists co-activating on router
└── docs/
    └── architecture.md             # design notes
```

## Quickstart

Requires Rust 1.75+ (`brew install rust` or `rustup`).

```bash
cargo test --workspace           # 15 tests pass

# Orchestration-only benchmarks on the deterministic stub backend
cargo run --release -p nrt-bench

# Real SLM inference smoke test — downloads TinyLlama 1.1B Q4 (~670 MB) on first run
cargo run --release -p nrt-backend-candle --example hello_candle

# Full cluster with real models + live dashboard at http://127.0.0.1:9000/
cargo run --release -p nrt-server -- \
  --manifest manifests/customer-support.yaml \
  --backend candle \
  --addr 127.0.0.1:9000

# ...or the stub backend if you want instant startup without a download:
cargo run --release -p nrt-server -- \
  --manifest manifests/customer-support.yaml \
  --backend stub \
  --addr 127.0.0.1:9000
```

Then in a second terminal:

```bash
curl -s http://127.0.0.1:9000/v1/cluster | jq
curl -s http://127.0.0.1:9000/v1/manifest | jq

curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"refund please"}],"max_tokens":8}' | jq
```

A live response looks like:

```json
{
  "id": "nrt-a9b583f2-...",
  "object": "chat.completion",
  "choices": [{ "index": 0, "message": { "role": "assistant",
                 "content": "[stub:technical] ok len=27 max=8" },
                 "finish_reason": "stop" }],
  "nrt": {
    "entry_model": "router",
    "final_model": "technical",
    "intent": "technical",
    "latency_ms": 102,
    "hops": [
      { "model": "router",    "completion": "intent=technical", "latency_ms": 47 },
      { "model": "technical", "completion": "[stub:technical] ok len=27 max=8", "latency_ms": 47 }
    ]
  }
}
```

## Tests

```
running 6 tests (nrt-manifest unit)
  dispatch::tests::parses_intent_dispatch ... ok
  dispatch::tests::parses_fixed_route ... ok
  dispatch::tests::rejects_missing_arrow ... ok
  dispatch::tests::rejects_malformed_lhs ... ok
  sizes::tests::parses_common_shapes ... ok
  sizes::tests::rejects_garbage ... ok

running 4 tests (nrt-manifest integration)
  parses_spec_example_exactly ... ok
  validation_surfaces_duplicate_ids ... ok
  validation_catches_unknown_entry ... ok
  loads_manifest_from_workspace_path ... ok

running 5 tests (nrt-cluster orchestration)
  cluster_bootstrap_places_models_at_declared_tiers ... ok
  chat_completion_dispatches_to_intent_specialist ... ok
  co_activation_warms_declared_models_after_router_fires ... ok
  unknown_intent_with_fallback_dispatches_to_fallback ... ok
  unknown_intent_without_fallback_returns_router_response ... ok

15 passed, 0 failed.
```

## Benchmark (stub backend, MacBook dev machine)

```
== customer-support (small, 5 models) ==
  bootstrap: 5 models, 3 resident, 1 standby, 2400 MB VRAM, 9 ms
  200 chat_completion calls: throughput 215 req/s
  latency: p50 4.65ms, p95 4.80ms, p99 4.84ms

== density-50 (stress, 51 models) ==
  bootstrap: 51 models, 26 resident, 25 standby, 20800 MB VRAM, 89 ms
  200 chat_completion calls: throughput 211 req/s
  latency: p50 4.65ms, p95 5.81ms, p99 5.86ms

== warming-heavy (co-activation stress, 9 models) ==
  bootstrap: 9 models, 1 resident, 8 standby, 800 MB VRAM, 15 ms
  200 chat_completion calls: throughput 216 req/s
  latency: p50 4.63ms, p95 4.70ms, p99 9.16ms
  co-activation warms fired: 18 (manifest declared 8 warmers)
```

What the benchmark is (and is not) measuring:

- **Measured**: orchestrator overhead — Manifest parse, tier state machine, dispatch resolve, session accounting, HTTP round-trip. Per-request overhead holds flat under 51 models because the registry is a `DashMap` (O(1) lookup) and tier transitions are amortized at bootstrap.
- **Not measured**: real inference throughput. The stub backend emits synthetic tokens; real numbers require llama.cpp / CUDA integration (Pillar 3 / Pillar 1 milestones).
- **Co-activation count under warming-heavy**: 18 warms over 200 requests. Expected floor is 8 (one per warmer); the extra 10 reflect fire-and-forget races during warmup — a second router hit arrives before a first-hit warm completes and the cluster double-fires `promote` on the still-Standby model. The backend handles double-promote gracefully. A future milestone adds per-model inflight-warm deduplication.

## Scope

This prototype addresses **Pillar 5** from the NRT spec (Manifests as first-class cluster objects) with enough of Pillar 1 to demonstrate the tier state machine. It is deliberately the smallest self-contained slice that proves:

1. The `nrt.yaml` format is fully parseable and round-trips through validation.
2. The Cluster Manager boots a cluster from a Manifest and places each model at its declared tier.
3. `router.output.intent -> specialists.{intent}` dispatch works end-to-end through a real HTTP endpoint.
4. Manifest-declared co-activation warming fires on router hit and moves Standby specialists to Resident without blocking the caller.
5. The test suite covers the failure modes (unknown intent, duplicate ids, malformed dispatch rule) the spec's Risk Register calls out.

Pillar 1 (real TMO on a live GPU with LRU KV eviction), Pillar 2 (hybrid-path NPU profiler), Pillar 3 (stream-load + JIT quantization), and Pillar 4 (per-session linear allocator) are each their own milestone — deliverable on top of this chassis.

## License

Apache 2.0.
