//! End-to-end tests that drive a cluster from a real Manifest fixture.
//!
//! These tests exercise the exact path a /v1/chat/completions call takes:
//! router inference -> dispatch resolution -> specialist inference, plus
//! the Manifest-declared co-activation warming fired from the router hit.

use nrt_backend_stub::{StubBackend, StubTiming};
use nrt_cluster::ClusterManager;
use nrt_core::{Backend, SessionId, Tier};
use nrt_manifest::load_from_str;
use std::{sync::Arc, time::Duration};

const SPEC_FIXTURE: &str = r#"
cluster: customer-support-agent
resources:
  gpu_budget: 18GB
  ram_budget: 32GB
models:
  router:
    source: hf://neurometric/router-1b-v2
    tier: resident
    priority: critical
    max_active_sessions: 16
  specialists:
    - id: billing
      source: hf://neurometric/billing-phi3
      tier: resident
      co_activation: router
    - id: technical
      source: hf://neurometric/tech-phi3
      tier: resident
      co_activation: router
    - id: legal
      source: hf://neurometric/legal-phi3
      tier: standby
      promotion_hint: on_router_classification
fallback:
  source: hf://anthropic/claude-haiku
  tier: remote
routing:
  entry: router
  dispatch_rule: router.output.intent -> specialists.{intent}
"#;

fn build_backend() -> Arc<dyn Backend> {
    let backend = StubBackend::new(StubTiming::instant());
    backend.register_router_intents(
        nrt_core::ModelId::new("router"),
        vec!["billing".to_string(), "technical".to_string(), "legal".to_string()],
    );
    Arc::new(backend)
}

#[tokio::test(flavor = "multi_thread")]
async fn cluster_bootstrap_places_models_at_declared_tiers() {
    let manifest = load_from_str(SPEC_FIXTURE).unwrap();
    let backend = build_backend();
    let cluster = ClusterManager::bootstrap(manifest, backend).await.unwrap();

    let snap = cluster.snapshot();
    let by_id = |name: &str| {
        snap.models
            .iter()
            .find(|m| m.id.as_str() == name)
            .unwrap_or_else(|| panic!("model {name} not in snapshot"))
    };
    assert_eq!(by_id("router").tier, Tier::Resident);
    assert_eq!(by_id("billing").tier, Tier::Resident);
    assert_eq!(by_id("technical").tier, Tier::Resident);
    assert_eq!(by_id("legal").tier, Tier::Standby);
    assert_eq!(by_id("fallback").tier, Tier::Remote);
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_completion_dispatches_to_intent_specialist() {
    let manifest = load_from_str(SPEC_FIXTURE).unwrap();
    let backend = build_backend();
    let cluster = ClusterManager::bootstrap(manifest, backend).await.unwrap();

    let result = cluster
        .chat_completion(Some(SessionId::new()), "the customer wants a refund".to_string(), 8)
        .await
        .unwrap();

    // The stub's djb2 hash routes this specific prompt deterministically.
    // The exact specialist depends on the hash; we only assert two invariants:
    //   1. There are two hops (router + specialist).
    //   2. The final model id is one of the declared specialists.
    assert_eq!(result.hops.len(), 2, "expected router + specialist hops");
    let final_id = result.final_model.as_str();
    assert!(
        matches!(final_id, "billing" | "technical" | "legal"),
        "final model {final_id} is not a known specialist"
    );
    // Router populates `intent`; specialist response does not.
    assert!(result.intent.is_some());
    assert_eq!(result.intent.as_deref().unwrap(), final_id);
}

#[tokio::test(flavor = "multi_thread")]
async fn co_activation_warms_declared_models_after_router_fires() {
    // Only `legal` is Standby in this manifest, but its co_activation is not the
    // router — its promotion_hint is. Let's redefine a tighter manifest where
    // exactly one Standby specialist declares co_activation: router.
    let yaml = r#"
cluster: warm-test
resources: { gpu_budget: 8GB, ram_budget: 16GB }
models:
  router: { source: stub://r, tier: resident }
  specialists:
    - { id: billing, source: stub://b, tier: resident, co_activation: router }
    - { id: technical, source: stub://t, tier: standby, co_activation: router }
routing:
  entry: router
  dispatch_rule: router.output.intent -> specialists.{intent}
"#;
    let manifest = load_from_str(yaml).unwrap();
    let backend = StubBackend::new(StubTiming::instant());
    backend.register_router_intents(
        nrt_core::ModelId::new("router"),
        vec!["billing".to_string(), "technical".to_string()],
    );
    let cluster = ClusterManager::bootstrap(manifest, Arc::new(backend))
        .await
        .unwrap();

    // Before any request: technical is Standby.
    let pre = cluster.snapshot();
    let technical_tier = pre
        .models
        .iter()
        .find(|m| m.id.as_str() == "technical")
        .unwrap()
        .tier;
    assert_eq!(technical_tier, Tier::Standby);

    // Fire a request. Router will hit, co-activation will warm technical.
    let _ = cluster
        .chat_completion(None, "any prompt".to_string(), 4)
        .await
        .unwrap();

    // Give the fire-and-forget warm task a moment to complete.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let post = cluster.snapshot();
    let technical_tier_after = post
        .models
        .iter()
        .find(|m| m.id.as_str() == "technical")
        .unwrap()
        .tier;
    assert_eq!(
        technical_tier_after,
        Tier::Resident,
        "technical should have been warmed to Resident via co-activation"
    );
    assert!(
        post.metrics.co_activation_warms >= 1,
        "expected >=1 co-activation warm, got {}",
        post.metrics.co_activation_warms
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_intent_with_fallback_dispatches_to_fallback() {
    // When router emits an intent the manifest doesn't know and a fallback is
    // declared, the cluster should dispatch to the fallback rather than erroring.
    let manifest = load_from_str(SPEC_FIXTURE).unwrap();
    let backend = StubBackend::new(StubTiming::instant());
    backend.register_router_intents(
        nrt_core::ModelId::new("router"),
        vec!["ghost".to_string()],
    );
    let cluster = ClusterManager::bootstrap(manifest, Arc::new(backend))
        .await
        .unwrap();

    let result = cluster
        .chat_completion(None, "bad prompt".to_string(), 4)
        .await
        .unwrap();
    assert_eq!(
        result.final_model.as_str(),
        "fallback",
        "unresolved intent with fallback declared should route to fallback"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_intent_without_fallback_returns_router_response() {
    // When no fallback is declared, the cluster falls back to the router's own
    // completion (preserves the response rather than erroring).
    let yaml = r#"
cluster: no-fallback
resources: { gpu_budget: 8GB, ram_budget: 16GB }
models:
  router: { source: stub://r, tier: resident }
  specialists:
    - { id: billing, source: stub://a, tier: resident }
routing:
  entry: router
  dispatch_rule: router.output.intent -> specialists.{intent}
"#;
    let manifest = load_from_str(yaml).unwrap();
    let backend = StubBackend::new(StubTiming::instant());
    backend.register_router_intents(
        nrt_core::ModelId::new("router"),
        vec!["ghost".to_string()],
    );
    let cluster = ClusterManager::bootstrap(manifest, Arc::new(backend))
        .await
        .unwrap();

    let result = cluster
        .chat_completion(None, "bad prompt".to_string(), 4)
        .await
        .unwrap();
    assert_eq!(result.final_model.as_str(), "router");
    assert_eq!(result.intent.as_deref(), Some("ghost"));
}
