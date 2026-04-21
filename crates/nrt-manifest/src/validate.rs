use crate::{dispatch::DispatchRule, Manifest, ManifestError};
use nrt_core::ModelId;
use std::collections::HashSet;

/// Structural validation result. The Cluster Manager applies additional
/// validation based on live resources (actual VRAM, available backends).
#[derive(Debug, Clone)]
pub struct ManifestValidation {
    pub known_ids: Vec<ModelId>,
    pub warnings: Vec<String>,
}

impl ManifestValidation {
    pub fn check(m: &Manifest) -> Result<Self, ManifestError> {
        let mut known: HashSet<ModelId> = HashSet::new();
        for r in m.all_models() {
            if !known.insert(r.id.clone()) {
                return Err(ManifestError::Invalid(format!(
                    "duplicate model id {:?}",
                    r.id.as_str()
                )));
            }
        }

        // The `entry` must resolve to a known id.
        if !known.contains(&m.routing.entry) {
            return Err(ManifestError::Invalid(format!(
                "routing.entry {:?} does not match any model id",
                m.routing.entry.as_str()
            )));
        }

        // Co-activation references must resolve.
        for r in m.all_models() {
            if let Some(co) = &r.co_activation {
                if !known.contains(co) {
                    return Err(ManifestError::Invalid(format!(
                        "model {:?} declares co_activation {:?} which is not a known model",
                        r.id.as_str(),
                        co.as_str()
                    )));
                }
            }
        }

        // Dispatch rule source must resolve.
        let mut warnings = Vec::new();
        match &m.routing.dispatch_rule {
            DispatchRule::IntentDispatch {
                source, target_set, ..
            } => {
                if !known.contains(source) {
                    return Err(ManifestError::Invalid(format!(
                        "dispatch rule source {:?} is not a known model",
                        source.as_str()
                    )));
                }
                if target_set != "specialists" {
                    warnings.push(format!(
                        "dispatch_rule target_set {target_set:?} is not 'specialists'; custom sets are not yet implemented"
                    ));
                }
            }
            DispatchRule::FixedRoute { source, target } => {
                if !known.contains(source) {
                    return Err(ManifestError::Invalid(format!(
                        "dispatch rule source {:?} is not a known model",
                        source.as_str()
                    )));
                }
                if !known.contains(target) {
                    return Err(ManifestError::Invalid(format!(
                        "dispatch rule target {:?} is not a known model",
                        target.as_str()
                    )));
                }
            }
        }

        // Soft check: resident footprint vs VRAM budget. Without real weight sizes
        // we can only warn — the cluster does the hard check at load time.
        let resident_count = m
            .all_models()
            .iter()
            .filter(|r| r.tier == nrt_core::Tier::Resident)
            .count();
        if resident_count > 0 && m.resources.gpu_budget_mb / (resident_count as u64) < 256 {
            warnings.push(format!(
                "gpu_budget {}MB / {resident_count} resident models = under 256 MB per model; density target (50+ models on 24GB) may not fit",
                m.resources.gpu_budget_mb
            ));
        }

        Ok(Self {
            known_ids: known.into_iter().collect(),
            warnings,
        })
    }
}
