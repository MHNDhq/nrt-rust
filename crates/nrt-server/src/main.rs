//! NRT HTTP server binary.
//!
//! Usage:
//!   nrt-server --manifest manifests/customer-support.yaml [--addr 127.0.0.1:9000]
//!
//! Endpoints:
//!   POST /v1/chat/completions  — OpenAI-compatible (subset)
//!   GET  /v1/cluster           — snapshot of tiers, sessions, metrics
//!   GET  /v1/manifest          — effective Manifest as JSON
//!   GET  /healthz              — liveness probe

use anyhow::Context;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use nrt_backend_candle::{router_profile, specialist_profile, CandleBackend, ModelMap};
use nrt_backend_stub::StubBackend;
use nrt_cluster::{ClusterHandle, ClusterManager, ClusterSnapshot, CompletionResult};
use nrt_core::{Backend, ModelId, SessionId};
use nrt_manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Clone)]
struct AppState {
    cluster: ClusterHandle,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let args: Args = Args::from_env();
    let manifest = nrt_manifest::load_from_path(&args.manifest_path)
        .with_context(|| format!("load manifest {}", args.manifest_path))?;
    tracing::info!(
        cluster = %manifest.cluster,
        models = manifest.all_models().len(),
        "manifest loaded"
    );

    let backend: Arc<dyn Backend> = match args.backend.as_str() {
        "stub" => build_stub_backend(&manifest),
        "candle" => build_candle_backend(&manifest).await?,
        other => anyhow::bail!(
            "unknown --backend {other:?}; expected one of: stub, candle"
        ),
    };
    let cluster = ClusterManager::bootstrap(manifest, backend).await?;

    let state = AppState { cluster };
    let app = Router::new()
        .route("/", get(dashboard))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/cluster", get(cluster_snapshot))
        .route("/v1/manifest", get(manifest_snapshot))
        .route("/healthz", get(healthz))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = args.addr.parse().context("parse --addr")?;
    tracing::info!(%addr, "starting nrt-server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .init();
}

/// Pre-populate the stub backend with intent sets inferred from the manifest:
/// the entry model is declared as a router, and the set of legal intents is the
/// list of specialist ids.
fn build_stub_backend(manifest: &Manifest) -> Arc<dyn Backend> {
    let backend = StubBackend::default();
    let intents: Vec<String> = manifest
        .models
        .specialists
        .iter()
        .map(|s| s.id.as_str().to_string())
        .collect();
    backend.register_router_intents(manifest.routing.entry.clone(), intents);
    Arc::new(backend)
}

/// Build the Candle backend with one profile per model in the manifest. Every
/// profile points at the same TinyLlama-1.1B Q4_K_M GGUF from HuggingFace;
/// the router and specialists differ only in their system prompt. This is the
/// pattern the NRT spec's "LoRA stacking for fine-tunes of a shared base"
/// approximates with real finetunes — here we approximate it with system prompts
/// so the prototype runs without training anything.
async fn build_candle_backend(manifest: &Manifest) -> anyhow::Result<Arc<dyn Backend>> {
    let intents: Vec<String> = manifest
        .models
        .specialists
        .iter()
        .map(|s| s.id.as_str().to_string())
        .collect();

    let mut profiles = ModelMap::new();
    profiles.insert(manifest.routing.entry.clone(), router_profile(intents.clone()));
    for s in &manifest.models.specialists {
        profiles.insert(s.id.clone(), specialist_profile(s.id.as_str()));
    }

    let backend = CandleBackend::new(profiles)?;
    // Remote-tier fallback (if declared): register a placeholder so the cluster
    // can load() it without fetching weights.
    if let Some(fb) = manifest.fallback.as_ref() {
        backend.register_remote(fb.id.clone());
    }
    // Pre-warm weights before cluster bootstrap so the first chat call doesn't
    // pay the download + tokenizer-load cost.
    let ids = backend.warm_all_registered().await?;
    tracing::info!(
        target: "nrt_server",
        warmed = ids.len(),
        ids = ?ids.iter().map(|i| i.as_str().to_string()).collect::<Vec<_>>(),
        "candle backend pre-warmed"
    );
    Ok(Arc::new(backend))
}

#[derive(Debug)]
struct Args {
    manifest_path: String,
    addr: String,
    backend: String,
}

impl Args {
    fn from_env() -> Self {
        let mut manifest_path = "manifests/customer-support.yaml".to_string();
        let mut addr = "127.0.0.1:9000".to_string();
        let mut backend = "stub".to_string();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--manifest" | "-m" => {
                    if let Some(v) = args.next() {
                        manifest_path = v;
                    }
                }
                "--addr" | "-a" => {
                    if let Some(v) = args.next() {
                        addr = v;
                    }
                }
                "--backend" | "-b" => {
                    if let Some(v) = args.next() {
                        backend = v;
                    }
                }
                "--help" | "-h" => {
                    println!("nrt-server [--manifest PATH] [--addr HOST:PORT] [--backend stub|candle]");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg {other:?}");
                    std::process::exit(2);
                }
            }
        }
        Self { manifest_path, addr, backend }
    }
}

//
// Handlers
//

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    session_id: Option<SessionId>,
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
}

fn default_max_tokens() -> u32 {
    32
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    choices: Vec<ChatChoice>,
    nrt: CompletionResult,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    content: String,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, AppError> {
    // Concatenate user messages as the prompt. Full ChatML conversion is a
    // future milestone — the prototype exercises the orchestration path.
    let prompt = req
        .messages
        .into_iter()
        .map(|m| m.content)
        .collect::<Vec<_>>()
        .join("\n");

    let result = state
        .cluster
        .chat_completion(req.session_id, prompt, req.max_tokens)
        .await
        .map_err(AppError::from)?;

    let resp = ChatCompletionResponse {
        id: format!("nrt-{}", result.session_id),
        object: "chat.completion",
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: result.completion.clone(),
            },
            finish_reason: "stop",
        }],
        nrt: result,
    };
    // Touch `req.model` to avoid dead_code warning; it's accepted but the
    // router-declared entry always wins in v0.
    let _ = req.model;
    Ok(Json(resp))
}

async fn cluster_snapshot(State(state): State<AppState>) -> Json<ClusterSnapshot> {
    Json(state.cluster.snapshot())
}

async fn manifest_snapshot(State(state): State<AppState>) -> Json<Manifest> {
    Json(state.cluster.manifest().clone())
}

async fn healthz() -> &'static str {
    "ok"
}

const DASHBOARD_HTML: &str = include_str!("../ui/dashboard.html");

async fn dashboard() -> axum::response::Html<&'static str> {
    axum::response::Html(DASHBOARD_HTML)
}

//
// Error plumbing
//

struct AppError(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.0.to_string(),
        });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}

// Silence unused-import warning when the binary is compiled on platforms that
// don't expose ModelId through the handler signatures.
#[allow(dead_code)]
fn _model_id_import_marker(_: &ModelId) {}
