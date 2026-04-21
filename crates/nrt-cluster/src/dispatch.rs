//! Dispatch: translate a router response into the next specialist invocation.
//!
//! The Manifest supplies a `DispatchRule`. The Cluster Manager calls
//! `dispatch::resolve` after the router returns, and uses the resolved
//! `ModelId` as the target of the specialist inference.

use nrt_core::{InferenceResponse, ModelId, NrtError, NrtResult};
use nrt_manifest::{DispatchRule, Manifest};

/// Resolve a router response to the next target model.
///
/// Returns `Ok(Some(id))` if the rule resolves to a known specialist.
/// Returns `Ok(None)` if the rule is a fixed route with no specialist in the
/// manifest (letting the caller fall through to the fallback policy).
/// Returns `Err(DispatchUnresolved)` if intent dispatch fires but the intent
/// does not match any known specialist.
pub fn resolve(
    manifest: &Manifest,
    router_output: &InferenceResponse,
) -> NrtResult<Option<ModelId>> {
    match &manifest.routing.dispatch_rule {
        DispatchRule::IntentDispatch { target_set, .. } => {
            // We only implement target_set = "specialists" in v0. Custom sets
            // are a documented future extension.
            if target_set != "specialists" {
                return Err(NrtError::backend(format!(
                    "dispatch target_set {target_set:?} not supported yet"
                )));
            }
            let Some(intent) = router_output.intent.as_ref() else {
                return Err(NrtError::DispatchUnresolved {
                    intent: "<none>".into(),
                    known: manifest
                        .models
                        .specialists
                        .iter()
                        .map(|s| s.id.as_str().to_string())
                        .collect(),
                });
            };
            let target = ModelId::new(intent);
            if manifest
                .models
                .specialists
                .iter()
                .any(|s| s.id == target)
            {
                Ok(Some(target))
            } else {
                Err(NrtError::DispatchUnresolved {
                    intent: intent.clone(),
                    known: manifest
                        .models
                        .specialists
                        .iter()
                        .map(|s| s.id.as_str().to_string())
                        .collect(),
                })
            }
        }
        DispatchRule::FixedRoute { target, .. } => {
            if manifest.find(target).is_some() {
                Ok(Some(target.clone()))
            } else {
                Ok(None)
            }
        }
    }
}
