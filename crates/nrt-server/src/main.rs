//! NRT HTTP server binary.
//!
//! Usage:
//!   nrt-server --manifest manifests/customer-support.yaml [--addr 127.0.0.1:9000]
//!
//! Endpoints:
//!   POST /v1/chat/completions  — OpenAI-shaped subset
//!   GET  /v1/cluster           — snapshot of tiers, sessions, metrics
//!   GET  /v1/manifest          — effective Manifest as JSON
//!   GET  /healthz              — liveness probe

use anyhow::Context;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::stream::{self, Stream};
use nrt_backend_candle::{router_profile, specialist_profile, CandleBackend, ModelMap};
use nrt_backend_stub::StubBackend;
use nrt_cluster::{
    ClusterHandle, ClusterManager, ClusterSnapshot, CompletionHop, CompletionResult,
};
use nrt_core::{Backend, ModelId, SessionId};
use nrt_manifest::Manifest;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
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
        "candle" => build_candle_backend(&manifest, CandleDevicePolicy::Auto).await?,
        "candle-metal" => build_candle_backend(&manifest, CandleDevicePolicy::ForceMetal).await?,
        other => anyhow::bail!(
            "unknown --backend {other:?}; expected one of: stub, candle, candle-metal"
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

/// Picked via `--backend candle` (Auto) vs `--backend candle-metal` (forced).
/// ForceMetal fails loudly if the binary wasn't compiled with the `metal`
/// feature or if Metal device 0 is unavailable, rather than silently
/// falling back to CPU.
#[derive(Debug, Clone, Copy)]
enum CandleDevicePolicy {
    Auto,
    ForceMetal,
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
async fn build_candle_backend(
    manifest: &Manifest,
    policy: CandleDevicePolicy,
) -> anyhow::Result<Arc<dyn Backend>> {
    let intents: Vec<String> = manifest
        .models
        .specialists
        .iter()
        .map(|s| s.id.as_str().to_string())
        .collect();

    let mut profiles = ModelMap::new();
    profiles.insert(
        manifest.routing.entry.clone(),
        router_profile(intents.clone()),
    );
    for s in &manifest.models.specialists {
        profiles.insert(s.id.clone(), specialist_profile(s.id.as_str()));
    }

    let backend = match policy {
        CandleDevicePolicy::Auto => CandleBackend::new(profiles)?,
        CandleDevicePolicy::ForceMetal => CandleBackend::new_metal(profiles)?,
    };
    tracing::info!(
        target: "nrt_server",
        device = %backend.device_kind(),
        policy = ?policy,
        "candle backend device"
    );
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
                    println!(
                        "nrt-server [--manifest PATH] [--addr HOST:PORT] \
                         [--backend stub|candle|candle-metal]"
                    );
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg {other:?}");
                    std::process::exit(2);
                }
            }
        }
        Self {
            manifest_path,
            addr,
            backend,
        }
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
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> u32 {
    32
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
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
    /// Populated for the router hop: the cluster emits a synthetic tool call
    /// of the form `dispatch_to_specialist(intent=...)` so the client sees the
    /// Manifest's dispatch rule as a function-call shape. Kept as an option so
    /// the final assistant message in non-router responses stays clean.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
}

/// OpenAI-compatible tool-call shape. NRT synthesizes these from the
/// Manifest's `dispatch_rule` rather than asking the model to emit them —
/// the spec's "router -> specialists.{intent}" rule is already a declarative
/// function call, so surfacing it here closes the OpenAI-API floor without
/// making the router do two jobs.
#[derive(Debug, Serialize, Clone)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ToolFunction {
    name: &'static str,
    arguments: String,
}

fn dispatch_tool_call(session_id: &SessionId, hop: &CompletionHop) -> Option<ToolCall> {
    let intent = hop.intent.as_ref()?;
    let args = serde_json::json!({
        "intent": intent,
        "target": format!("specialists.{}", intent),
    });
    Some(ToolCall {
        id: format!("call_{}_{}", session_id, hop.model.as_str()),
        kind: "function",
        function: ToolFunction {
            name: "dispatch_to_specialist",
            arguments: args.to_string(),
        },
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, AppError> {
    validate_chat_request(&state.cluster, &req)?;
    let streaming = req.stream;
    let prompt = format_chat_prompt(&req.messages);

    let result = state
        .cluster
        .chat_completion(req.session_id, prompt, req.max_tokens)
        .await
        .map_err(AppError::from)?;

    if streaming {
        Ok(chat_completions_sse(result).into_response())
    } else {
        Ok(chat_completions_json(result).into_response())
    }
}

fn chat_completions_json(result: CompletionResult) -> Json<ChatCompletionResponse> {
    let router_hop = result.hops.first();
    let tool_calls = router_hop
        .and_then(|hop| dispatch_tool_call(&result.session_id, hop))
        .into_iter()
        .collect::<Vec<_>>();

    let resp = ChatCompletionResponse {
        id: format!("nrt-{}", result.session_id),
        object: "chat.completion",
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: result.completion.clone(),
                tool_calls,
            },
            finish_reason: "stop",
        }],
        nrt: result,
    };
    Json(resp)
}

/// OpenAI-shaped Server-Sent Events. We don't yet have token-level streaming
/// out of the Candle backend (that's an M2 item — per-token yield through the
/// Cluster Manager's hop boundary), so the current implementation streams at
/// *hop granularity*: one SSE frame per completed hop (router, then
/// specialist), plus a terminating `[DONE]` sentinel. Hop-level streaming
/// already buys the client a progressive UX — a user sees the router's
/// classification decision before the specialist finishes composing.
/// Token-level streaming swaps the inner generator for a `mpsc::Receiver<Token>`
/// without changing this wire format.
fn chat_completions_sse(
    result: CompletionResult,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session = result.session_id;
    let id = format!("nrt-{session}");
    let final_model = result.final_model.clone();

    let mut frames: Vec<Result<Event, Infallible>> = Vec::new();

    for (idx, hop) in result.hops.iter().enumerate() {
        let tool_calls = dispatch_tool_call(&session, hop)
            .into_iter()
            .collect::<Vec<_>>();
        // OpenAI shape: when tool_calls are present, content is empty on that
        // delta. The raw per-hop completion still reaches clients through the
        // non-standard `nrt.hop_completion` field so demos and dashboards can
        // render both the tool call and the router's raw output.
        let delta = if tool_calls.is_empty() {
            serde_json::json!({
                "role": "assistant",
                "content": hop.completion,
            })
        } else {
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            })
        };
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": serde_json::Value::Null,
            }],
            "nrt": {
                "hop_index": idx,
                "model": hop.model,
                "intent": hop.intent,
                "latency_ms": hop.latency_ms,
                "hop_completion": hop.completion,
            },
        });
        frames.push(Ok(Event::default().data(chunk.to_string())));
    }

    let closing = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "nrt": {
            "final_model": final_model,
            "latency_ms": result.latency_ms,
            "hops_total": result.hops.len(),
        },
    });
    frames.push(Ok(Event::default().data(closing.to_string())));
    frames.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(frames)).keep_alive(KeepAlive::default())
}

async fn cluster_snapshot(State(state): State<AppState>) -> Json<ClusterSnapshot> {
    Json(state.cluster.snapshot())
}

async fn manifest_snapshot(State(state): State<AppState>) -> Json<Manifest> {
    Json(state.cluster.manifest())
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

fn validate_chat_request(
    cluster: &ClusterHandle,
    req: &ChatCompletionRequest,
) -> Result<(), AppError> {
    if req.messages.is_empty() {
        return Err(AppError::bad_request(
            "messages must contain at least one chat message",
        ));
    }
    validate_requested_model(req.model.as_deref(), &cluster.manifest().routing.entry)
}

fn validate_requested_model(requested: Option<&str>, entry: &ModelId) -> Result<(), AppError> {
    if let Some(model) = requested {
        if model != entry.as_str() {
            return Err(AppError::bad_request(format!(
                "model {:?} is not supported by this routed endpoint; use {:?}",
                model,
                entry.as_str()
            )));
        }
    }
    Ok(())
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| {
            format!(
                "<|{}|>\n{}",
                normalize_role(&message.role),
                message.content.trim_end()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_role(role: &str) -> &'static str {
    if role.eq_ignore_ascii_case("system") {
        "system"
    } else if role.eq_ignore_ascii_case("assistant") {
        "assistant"
    } else if role.eq_ignore_ascii_case("tool") {
        "tool"
    } else {
        "user"
    }
}

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.into().to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.message,
        });
        (self.status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_model_must_match_manifest_entry() {
        let entry = ModelId::new("router");
        assert!(validate_requested_model(Some("router"), &entry).is_ok());
        assert!(validate_requested_model(None, &entry).is_ok());
        assert!(validate_requested_model(Some("billing"), &entry).is_err());
    }

    #[test]
    fn prompt_format_preserves_role_boundaries() {
        let prompt = format_chat_prompt(&[
            ChatMessage {
                role: "system".into(),
                content: "be concise".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "refund please".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "What order number?".into(),
            },
        ]);

        assert!(prompt.contains("<|system|>\nbe concise"));
        assert!(prompt.contains("<|user|>\nrefund please"));
        assert!(prompt.contains("<|assistant|>\nWhat order number?"));
    }

    #[test]
    fn dispatch_tool_call_uses_manifest_rule_shape() {
        let session = SessionId::new();
        let router_hop = CompletionHop {
            model: ModelId::new("router"),
            completion: "intent=billing".into(),
            intent: Some("billing".into()),
            latency_ms: 47,
        };
        let tc = dispatch_tool_call(&session, &router_hop).expect("intent present -> tool call");
        assert_eq!(tc.kind, "function");
        assert_eq!(tc.function.name, "dispatch_to_specialist");
        // The args are a JSON object; parse them back to assert shape.
        let parsed: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(parsed["intent"], "billing");
        assert_eq!(parsed["target"], "specialists.billing");
    }

    #[test]
    fn dispatch_tool_call_skipped_when_intent_absent() {
        let session = SessionId::new();
        let specialist_hop = CompletionHop {
            model: ModelId::new("billing"),
            completion: "we will refund you".into(),
            intent: None,
            latency_ms: 320,
        };
        assert!(dispatch_tool_call(&session, &specialist_hop).is_none());
    }

    #[test]
    fn non_streaming_response_surfaces_tool_call_on_router_hop() {
        let session = SessionId::new();
        let result = CompletionResult {
            session_id: session,
            entry_model: ModelId::new("router"),
            final_model: ModelId::new("billing"),
            intent: Some("billing".into()),
            completion: "we will refund you".into(),
            latency_ms: 367,
            hops: vec![
                CompletionHop {
                    model: ModelId::new("router"),
                    completion: "intent=billing".into(),
                    intent: Some("billing".into()),
                    latency_ms: 47,
                },
                CompletionHop {
                    model: ModelId::new("billing"),
                    completion: "we will refund you".into(),
                    intent: None,
                    latency_ms: 320,
                },
            ],
        };
        let Json(resp) = chat_completions_json(result);
        assert_eq!(resp.choices.len(), 1);
        let choice = &resp.choices[0];
        assert_eq!(choice.message.tool_calls.len(), 1);
        assert_eq!(
            choice.message.tool_calls[0].function.name,
            "dispatch_to_specialist"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&choice.message.tool_calls[0].function.arguments).unwrap();
        assert_eq!(parsed["intent"], "billing");
    }
}
