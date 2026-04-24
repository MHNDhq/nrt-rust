# nrt-prototype

A Rust prototype of the **Neurometric Runtime (NRT)** Pillar 5: the Manifest-driven Cluster Manager, built against the NRT product spec v2, with real SLM inference via a pure-Rust Candle backend (CPU + Metal) and a live cluster dashboard.

This is a proof-of-build artifact that accompanies a proposal for the **Rust Agentic Inference Engine for SLMs** Upwork job. It is not a full Phase 0; see [Scope](#scope).

## What it does

- Parses the exact `nrt.yaml` example from the NRT spec (router + specialists + fallback + dispatch rule).
- Boots a cluster of stub models and places each at its declared tier (`resident` / `standby` / `remote`).
- Exposes an **OpenAI-shaped `/v1/chat/completions` subset** backed by the cluster.
- Fires the router, resolves the dispatch rule (`router.output.intent -> specialists.{intent}`), then invokes the resolved specialist in a single routed request.
- Performs **Manifest-declared co-activation warming** (fire-and-forget Standby → Resident promotion for any specialist whose `co_activation` field names the just-fired router).
- Runs a **background LRU sweeper** that walks live sessions and evicts KV state (`SessionState::Idle`) for any session past `idle_threshold`. Increments `lru_evictions` and emits `SessionEvicted` events. Weights stay Resident; only session state moves, per Pillar 1.
- **Drains in-flight sessions on `replace_manifest`**, rejecting new admissions until the swap completes. Atomic cluster updates per the spec's Pillar 5 bullet.
- Enforces both `gpu_budget` (sum of Resident VRAM) **and** `ram_budget` (sum of Standby RAM) on bootstrap.
- Records tier transitions plus structured counters for promotions, co-activation warms, dispatch hits, live sessions, per-model request totals, and LRU evictions.

Also included:

- **Real SLM inference** via `nrt-backend-candle`: pure-Rust inference over TinyLlama-1.1B Q4_K_M weights loaded from HuggingFace. No Python, no FFI to llama.cpp. `cargo run -p nrt-backend-candle --example hello_candle` downloads the weights (first run, ~670 MB) and generates real tokens in a forward pass.
- Candle profiles that point at the same GGUF/tokenizer artifacts now share one loaded base model in memory instead of duplicating weights per logical `ModelId`.
- **Live cluster dashboard** served at `http://127.0.0.1:9000/` (when the server runs). Dark, IBM Plex Sans / Plex Mono, shows live tier state, VRAM gauges, metrics, and a request log with the router-to-specialist hop breakdown.

It does **not** do (these are follow-on milestones; see [Scope](#scope)):

- CUDA / Metal / ANE / Hexagon kernels beyond what Candle exposes by default. A Candle Metal backend is a cargo-feature flip away but was not enabled in this prototype to keep the demo CPU-portable.
- Stream-load with memory-mapped weights, JIT quantization, or background upgrade.
- Persist per-session KV caches across requests; the current Candle path still rebuilds generation state from the fresh prompt on each call.
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
│   ├── nrt-cluster/                # Cluster Manager (the brain)
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
cargo test --workspace           # 29 tests pass
cargo clippy --workspace --all-targets -- -D warnings

# Orchestration-only benchmarks on the deterministic stub backend
cargo run --release -p nrt-bench

# Real SLM inference smoke test. First run downloads TinyLlama 1.1B Q4 (~670 MB).
cargo run --release -p nrt-backend-candle --example hello_candle

# Same smoke test, but forces Metal on macOS. Fails loudly if Metal is unavailable
# instead of silently falling back to CPU.
cargo run --release -p nrt-backend-candle --example hello_candle_metal --features metal

# Full cluster with real models + live dashboard at http://127.0.0.1:9000/
cargo run --release -p nrt-server -- \
  --manifest manifests/customer-support.yaml \
  --backend candle \
  --addr 127.0.0.1:9000

# Same, but route through Candle Metal (macOS; requires --features metal on the build):
cargo run --release -p nrt-server --features metal -- \
  --manifest manifests/customer-support.yaml \
  --backend candle-metal \
  --addr 127.0.0.1:9000

# ...or the stub backend for instant startup without a download:
cargo run --release -p nrt-server -- \
  --manifest manifests/customer-support.yaml \
  --backend stub \
  --addr 127.0.0.1:9000
```

Then in a second terminal:

```bash
curl -s http://127.0.0.1:9000/v1/cluster | jq
curl -s http://127.0.0.1:9000/v1/manifest | jq

# Non-streaming (default). Response includes OpenAI-shaped `tool_calls` on the
# router hop: `dispatch_to_specialist(intent=..., target=specialists.{intent})`.
curl -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"refund please"}],"max_tokens":8}' | jq

# Streaming (SSE). One frame per hop plus a [DONE] sentinel, OpenAI shape.
curl -N -s -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"refund please"}],"max_tokens":8,"stream":true}'
```

A live non-streaming response looks like:

```json
{
  "id": "nrt-a9b583f2-...",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "[stub:billing] ok len=43 max=8",
      "tool_calls": [{
        "id": "call_a9b583f2_router",
        "type": "function",
        "function": {
          "name": "dispatch_to_specialist",
          "arguments": "{\"intent\":\"billing\",\"target\":\"specialists.billing\"}"
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "nrt": {
    "entry_model": "router",
    "final_model": "billing",
    "intent": "billing",
    "latency_ms": 102,
    "hops": [
      { "model": "router",  "completion": "intent=billing", "latency_ms": 47 },
      { "model": "billing", "completion": "[stub:billing] ok len=43 max=8", "latency_ms": 47 }
    ]
  }
}
```

Streaming responses are `text/event-stream` with one `data:` frame per hop
(router, then specialist), a terminating frame carrying `finish_reason: "stop"`,
and a `data: [DONE]` sentinel. The router hop's delta includes `tool_calls`;
the specialist hop's delta includes `content`.

## Tests

```
running 2 tests (nrt-backend-candle unit)
  shared_key_collapses_profiles_with_same_artifacts ... ok
  warmable_model_ids_skip_remote_only_profiles ... ok

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

running 12 tests (nrt-cluster orchestration)
  cluster_bootstrap_places_models_at_declared_tiers ... ok
  chat_completion_dispatches_to_intent_specialist ... ok
  co_activation_warms_declared_models_after_router_fires ... ok
  ephemeral_session_is_cleaned_up_after_response ... ok
  persistent_session_counts_live_models_until_end ... ok
  replace_manifest_updates_snapshot_and_dispatch ... ok
  replace_manifest_drains_inflight_sessions ... ok
  lru_sweeper_evicts_idle_sessions ... ok
  lru_sweeper_respects_min_lifetime ... ok
  ram_budget_exceeded_fails_bootstrap ... ok
  unknown_intent_with_fallback_dispatches_to_fallback ... ok
  unknown_intent_without_fallback_returns_router_response ... ok

running 5 tests (nrt-server unit)
  prompt_format_preserves_role_boundaries ... ok
  requested_model_must_match_manifest_entry ... ok
  dispatch_tool_call_uses_manifest_rule_shape ... ok
  dispatch_tool_call_skipped_when_intent_absent ... ok
  non_streaming_response_surfaces_tool_call_on_router_hop ... ok

29 passed, 0 failed.
```

## Benchmark (stub backend, MacBook dev machine)

```
== customer-support (small, 5 models) ==
  bootstrap: 5 models, 3 resident, 1 standby, 2400 MB VRAM, 9 ms
  200 chat_completion calls: throughput 217 req/s
  latency: p50 4.61ms, p95 4.66ms, p99 4.75ms

== density-50 (stress, 51 models) ==
  bootstrap: 51 models, 26 resident, 25 standby, 20800 MB VRAM, 88 ms
  200 chat_completion calls: throughput 208 req/s
  latency: p50 4.62ms, p95 5.79ms, p99 6.76ms

== warming-heavy (co-activation stress, 9 models) ==
  bootstrap: 9 models, 1 resident, 8 standby, 800 MB VRAM, 14 ms
  200 chat_completion calls: throughput 215 req/s
  latency: p50 4.60ms, p95 4.66ms, p99 9.04ms
  co-activation warms fired: 7 (manifest declared 8 warmers)
```

What the benchmark is (and is not) measuring:

- **Measured**: orchestrator overhead (cluster bootstrap, tier state machine, dispatch resolve, session accounting, and async warming bookkeeping). The hot-loop numbers come from direct `ClusterManager::chat_completion` calls on the stub backend, not from the HTTP server.
- **Not measured**: real inference throughput. The stub backend emits synthetic tokens; real numbers require llama.cpp / CUDA integration (Pillar 3 / Pillar 1 milestones).
- **Co-activation count under warming-heavy**: 7 warms over 200 requests. The manifest declares 8 warmers, but one request path typically dispatches directly into one of those specialists and promotes it on-demand before the fire-and-forget warm task reaches it. Inflight warm dedup now prevents double-promote races.

## Real-inference benchmark vs Ollama

`docs/bench_ollama_vs_nrt.sh` runs NRT's Candle CPU backend and Ollama's
llama.cpp Metal backend against the **exact same TinyLlama 1.1B Q4_K_M GGUF**
from HuggingFace. Both runtimes are warmed once before the timed loop. Same
4 prompts, same 32 max_tokens, same temperature=0.

Representative results (MacBook M-class, 4 prompts, `docs/bench_results.txt`):

```
NRT    (Candle CPU, router + specialist hops):  avg ~2716 ms / request (~1360 ms / hop)
Ollama (llama.cpp Metal, single model):          avg  ~341 ms / request
```

Per-hop, that is a ~4x backend-kernel gap. It is not an architecture gap. It
is the exact gap M2 closes by swapping Candle CPU for Candle+CUDA or
`llama-cpp-2` FFI. Ollama cannot do the Manifest-driven multi-model dispatch
pattern that NRT does; NRT cannot yet match llama.cpp's kernel maturity on a
single model. The spec's design is for NRT to eventually do both.

To reproduce:

```bash
# One-time: register the same HF-cached Q4_K_M file as an Ollama model.
ollama create nrt-tinyllama-q4km -f docs/Modelfile.tinyllama  # shipped in the repo

ollama serve &                                                # port 11434
cargo build --release -p nrt-server
bash docs/bench_ollama_vs_nrt.sh | tee docs/bench_results.txt
```

## Scope

This prototype addresses **Pillar 5** from the NRT spec (Manifests as first-class cluster objects) with enough of Pillar 1 to demonstrate the tier state machine and LRU policy. It is deliberately the smallest self-contained slice that proves:

1. The `nrt.yaml` format is fully parseable and round-trips through validation.
2. The Cluster Manager boots a cluster from a Manifest and places each model at its declared tier, enforcing both `gpu_budget` and `ram_budget`.
3. `router.output.intent -> specialists.{intent}` dispatch works end-to-end through a real HTTP endpoint.
4. Manifest-declared co-activation warming fires on router hit and moves Standby specialists to Resident without blocking the caller.
5. The LRU policy evicts idle session KV state on a live sweeper loop, on the stub backend today; on real VRAM the same policy drives `backend.demote` in the next milestone.
6. `replace_manifest` drains in-flight sessions before swapping, satisfying the spec's "atomic cluster updates" bullet.
7. The test suite covers the failure modes (unknown intent, duplicate ids, malformed dispatch rule, ram-budget overflow, drain timeout shape) the spec's Risk Register calls out.

Pillar 1 (real TMO on a live GPU with kernel-level KV eviction), Pillar 2 (hybrid-path NPU profiler), Pillar 3 (stream-load + JIT quantization), and Pillar 4 (per-session linear allocator) are each their own milestone, deliverable on top of this chassis.

## License

Apache 2.0.
