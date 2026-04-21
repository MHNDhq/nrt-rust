//! Integration tests for the full YAML -> Manifest pipeline.

use nrt_core::{ModelId, Tier};
use nrt_manifest::{load_from_path, load_from_str, DispatchRule, Priority};

const SPEC_FIXTURE: &str = include_str!("fixtures/customer-support.yaml");

#[test]
fn parses_spec_example_exactly() {
    let m = load_from_str(SPEC_FIXTURE).expect("spec fixture must parse");
    assert_eq!(m.cluster, "customer-support-agent");
    assert_eq!(m.resources.gpu_budget_mb, 18 * 1024);
    assert_eq!(m.resources.ram_budget_mb, 32 * 1024);

    // Router fields
    assert_eq!(m.models.router.id.as_str(), "router");
    assert_eq!(m.models.router.source, "hf://neurometric/router-1b-v2");
    assert_eq!(m.models.router.tier, Tier::Resident);
    assert_eq!(m.models.router.priority, Priority::Critical);
    assert_eq!(m.models.router.max_active_sessions, Some(16));

    // Specialists
    assert_eq!(m.models.specialists.len(), 3);
    let legal = m
        .models
        .specialists
        .iter()
        .find(|s| s.id.as_str() == "legal")
        .expect("legal specialist");
    assert_eq!(legal.tier, Tier::Standby);
    assert_eq!(
        legal.promotion_hint.as_deref(),
        Some("on_router_classification")
    );

    let billing = m
        .models
        .specialists
        .iter()
        .find(|s| s.id.as_str() == "billing")
        .expect("billing specialist");
    assert_eq!(billing.co_activation, Some(ModelId::new("router")));

    // Fallback
    let fb = m.fallback.as_ref().expect("fallback present");
    assert_eq!(fb.id.as_str(), "fallback");
    assert_eq!(fb.tier, Tier::Remote);

    // Dispatch rule
    match &m.routing.dispatch_rule {
        DispatchRule::IntentDispatch {
            source,
            field,
            target_set,
        } => {
            assert_eq!(source.as_str(), "router");
            assert_eq!(field, "intent");
            assert_eq!(target_set, "specialists");
        }
        _ => panic!("expected IntentDispatch"),
    }
}

#[test]
fn validation_surfaces_duplicate_ids() {
    let yaml = r#"
cluster: bad
resources: { gpu_budget: 4GB, ram_budget: 8GB }
models:
  router: { source: stub://r, tier: resident }
  specialists:
    - { id: billing, source: stub://a, tier: resident }
    - { id: billing, source: stub://b, tier: resident }
routing:
  entry: router
  dispatch_rule: router.output.intent -> specialists.{intent}
"#;
    let err = load_from_str(yaml).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate"),
        "expected duplicate error, got {msg}"
    );
}

#[test]
fn validation_catches_unknown_entry() {
    let yaml = r#"
cluster: bad-entry
resources: { gpu_budget: 4GB, ram_budget: 8GB }
models:
  router: { source: stub://r, tier: resident }
  specialists:
    - { id: billing, source: stub://a, tier: resident }
routing:
  entry: ghost
  dispatch_rule: router.output.intent -> specialists.{intent}
"#;
    let err = load_from_str(yaml).unwrap_err();
    assert!(err.to_string().contains("routing.entry"));
}

#[test]
fn loads_manifest_from_workspace_path() {
    // Exercises the on-disk path used by the nrt-server binary.
    let here = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(here)
        .parent() // crates
        .unwrap()
        .parent() // nrt-prototype
        .unwrap()
        .join("manifests")
        .join("customer-support.yaml");
    let m = load_from_path(&path).unwrap_or_else(|e| panic!("load {path:?}: {e}"));
    assert_eq!(m.cluster, "customer-support-agent");
}
