//! Smoke test for the Candle backend: load a TinyLlama 1.1B Q4_K_M GGUF and
//! run one completion. First run downloads ~670 MB; subsequent runs hit cache.
//!
//!   cargo run --release -p nrt-backend-candle --example hello_candle

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

    let model_id = ModelId::new("tinyllama");
    let profiles = ModelMap::new().with(model_id.clone(), default_tinyllama_profile());
    let backend = CandleBackend::new(profiles)?;

    println!("[1/3] Downloading + loading weights (first run takes ~60s for ~670 MB)...");
    let t0 = Instant::now();
    let handle = backend.load(&model_id, Tier::Resident).await?;
    println!(
        "      loaded in {:.2}s  [tier={:?} vram_mb={}]",
        t0.elapsed().as_secs_f32(),
        handle.tier,
        handle.resident_vram_mb
    );

    let prompt = "In one short sentence, what does an SLM inference runtime do?";
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
                max_tokens: 64,
                temperature: None,
                extra: Default::default(),
            },
        )
        .await?;
    let elapsed = t0.elapsed();
    println!("[3/3] generated {} tokens in {:.2}s:", response.tokens_emitted, elapsed.as_secs_f32());
    println!("      {}", response.completion.trim());
    println!(
        "      throughput: {:.1} tok/s",
        response.tokens_emitted as f64 / elapsed.as_secs_f64().max(0.001)
    );

    Ok(())
}
