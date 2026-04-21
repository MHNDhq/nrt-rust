//! NRT benchmark harness.
//!
//! Measures three numbers called out in the NRT spec's "Phase 0" success metrics:
//!   1. Density: resident model count under a VRAM budget
//!   2. Co-activation warm cost: latency saved by Manifest-declared pre-warming
//!   3. Cluster orchestrator overhead: μs per request above backend cost
//!
//! The numbers are meaningful for orchestrator overhead only — the backend is
//! the stub, so absolute inference latency reflects the stub's configured
//! timing profile, not real model throughput. This is exactly what we want
//! to benchmark: everything above the backend.

use nrt_backend_stub::{StubBackend, StubTiming};
use nrt_cluster::{settle, ClusterManager};
use nrt_core::{Backend, SessionId};
use nrt_manifest::load_from_path;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

const SUITES: &[(&str, &str)] = &[
    (
        "customer-support (small, 5 models)",
        "manifests/customer-support.yaml",
    ),
    (
        "density-50 (stress, 51 models)",
        "manifests/density-50.yaml",
    ),
    (
        "warming-heavy (co-activation stress, 9 models)",
        "manifests/warming-heavy.yaml",
    ),
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    println!("NRT benchmark harness — orchestrator overhead only (stub backend)\n");

    for (name, path) in SUITES {
        println!("== {name} ==");
        run_suite(path).await?;
        println!();
    }
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
}

async fn run_suite(manifest_path_str: &str) -> anyhow::Result<()> {
    let manifest_path = project_root().join(manifest_path_str);
    let manifest = load_from_path(&manifest_path)?;

    // 1. Density — how many models got Resident on startup.
    let backend = build_stub_backend_with_instant_timing(&manifest);
    let boot_start = Instant::now();
    let cluster = ClusterManager::bootstrap(manifest.clone(), backend.clone()).await?;
    let boot_ms = boot_start.elapsed().as_millis();
    let snap = cluster.snapshot();
    let resident = snap
        .models
        .iter()
        .filter(|m| m.tier == nrt_core::Tier::Resident)
        .count();
    let standby = snap
        .models
        .iter()
        .filter(|m| m.tier == nrt_core::Tier::Standby)
        .count();
    let vram_mb: u64 = snap.models.iter().map(|m| m.resident_vram_mb).sum();
    println!(
        "  bootstrap: {} models, {} resident, {} standby, {} MB VRAM, {} ms",
        snap.models.len(),
        resident,
        standby,
        vram_mb,
        boot_ms
    );

    // 2. Orchestrator overhead — fire N requests, measure end-to-end per request.
    let reqs = 200u32;
    let mut latencies = Vec::with_capacity(reqs as usize);
    let bench_start = Instant::now();
    for i in 0..reqs {
        let t0 = Instant::now();
        let res = cluster
            .chat_completion(Some(SessionId::new()), format!("benchmark-prompt #{i}"), 8)
            .await?;
        latencies.push(t0.elapsed());
        std::hint::black_box(&res);
    }
    settle(&cluster).await;
    let total = bench_start.elapsed();
    let (p50, p95, p99) = percentiles(&mut latencies);
    println!(
        "  {reqs} chat_completion calls: total {:.2}s, throughput {:.0} req/s",
        total.as_secs_f64(),
        reqs as f64 / total.as_secs_f64()
    );
    println!(
        "  latency: p50 {:.2}ms, p95 {:.2}ms, p99 {:.2}ms (backend: instant stub)",
        to_ms(p50),
        to_ms(p95),
        to_ms(p99)
    );

    // 3. Co-activation warming shape.
    let metrics = snap.metrics.clone();
    let new_metrics = cluster.snapshot().metrics;
    let warms = new_metrics
        .co_activation_warms
        .saturating_sub(metrics.co_activation_warms);
    println!(
        "  co-activation warms fired across benchmark: {warms} (manifest declared {} warmers)",
        manifest
            .all_models()
            .iter()
            .filter(|m| m.co_activation.is_some())
            .count()
    );

    Ok(())
}

fn percentiles(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort();
    let p = |f: f64| -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }
        let i = ((samples.len() as f64) * f).floor() as usize;
        samples[i.min(samples.len() - 1)]
    };
    (p(0.5), p(0.95), p(0.99))
}

fn to_ms(d: Duration) -> f64 {
    (d.as_micros() as f64) / 1000.0
}

fn build_stub_backend_with_instant_timing(manifest: &nrt_manifest::Manifest) -> Arc<dyn Backend> {
    // For the warming-heavy suite we want promotions to take non-zero time so
    // the benchmark can actually measure the pipeline — otherwise the fire-and-
    // forget warms complete before the main chat_completion returns and the
    // counter reads zero even though the behavior is correct.
    let timing = if manifest.cluster.contains("warming") {
        StubTiming {
            load_to_standby_ms: 0,
            promote_to_resident_ms: 3,
            kv_evict_ms: 0,
            kv_restore_ms: 0,
            per_token_ms: 0,
            prefill_ms: 0,
        }
    } else {
        StubTiming::instant()
    };
    let backend = StubBackend::new(timing);
    let intents: Vec<String> = manifest
        .models
        .specialists
        .iter()
        .map(|s| s.id.as_str().to_string())
        .collect();
    backend.register_router_intents(manifest.routing.entry.clone(), intents);
    Arc::new(backend)
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}
