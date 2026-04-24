//! Smoke test for the Candle Metal path. Forces Device::new_metal(0) and
//! errors loudly if Metal is unavailable rather than silently falling back
//! to CPU. Use this to prove the `metal` feature is wired and working on
//! the reviewer's Apple Silicon machine.
//!
//!   cargo run --release -p nrt-backend-candle \
//!     --example hello_candle_metal \
//!     --features metal
//!
//! First run downloads TinyLlama 1.1B Q4_K_M (~670 MB); subsequent runs hit
//! the HuggingFace cache.

use nrt_backend_candle::{default_tinyllama_profile, CandleBackend, ModelMap};
use nrt_core::{Backend, InferenceRequest, ModelId, SessionId, Tier};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let model_id = ModelId::new("tinyllama-metal");
    let profiles = ModelMap::new().with(model_id.clone(), default_tinyllama_profile());

    println!("[0/3] Forcing Candle device = Metal(0) ...");
    let backend = CandleBackend::new_metal(profiles).map_err(|e| {
        eprintln!(
            "Metal backend build failed: {e}\n\
             Hint: rebuild with `--features metal` on macOS 13+."
        );
        e
    })?;
    println!("      device = {}", backend.device_kind());

    println!("[1/3] Downloading + loading weights (first run: ~60s for ~670 MB)...");
    let t0 = Instant::now();
    let handle = backend.load(&model_id, Tier::Resident).await?;
    println!(
        "      loaded in {:.2}s  [tier={:?} vram_mb={}]",
        t0.elapsed().as_secs_f32(),
        handle.tier,
        handle.resident_vram_mb
    );

    let prompt = "In one short sentence, what is the Apple Neural Engine?";
    println!("[2/3] prompt: {prompt:?}");

    let t0 = Instant::now();
    let response = backend
        .infer(
            &handle,
            None,
            InferenceRequest {
                session_id: SessionId::new(),
                model_id: model_id.clone(),
                prompt: prompt.to_string(),
                max_tokens: 48,
                temperature: None,
                extra: Default::default(),
            },
        )
        .await?;
    let elapsed = t0.elapsed();
    println!(
        "[3/3] generated {} tokens in {:.2}s on {} backend:",
        response.tokens_emitted,
        elapsed.as_secs_f32(),
        backend.device_kind()
    );
    println!("      {}", response.completion.trim());
    println!(
        "      throughput: {:.1} tok/s",
        response.tokens_emitted as f64 / elapsed.as_secs_f64().max(0.001)
    );

    Ok(())
}
