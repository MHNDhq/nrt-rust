//! Parser and validator for the NRT Manifest YAML format.
//!
//! Target format reproduces the example from the NRT product spec v2 exactly:
//!
//! ```yaml
//! cluster: customer-support-agent
//! resources:
//!   gpu_budget: 18GB
//!   ram_budget: 32GB
//! models:
//!   router:
//!     source: hf://neurometric/router-1b-v2
//!     tier: resident
//!     priority: critical
//!     max_active_sessions: 16
//!   specialists:
//!     - id: billing
//!       source: hf://neurometric/billing-phi3
//!       tier: resident
//!       co_activation: router
//! fallback:
//!   source: hf://anthropic/claude-haiku
//!   tier: remote
//! routing:
//!   entry: router
//!   dispatch_rule: router.output.intent -> specialists.{intent}
//! ```
//!
//! The parser is intentionally permissive on tier casing and size suffix casing
//! (`GB`/`gb`/`Gb` all parse) but strict on unknown top-level keys — typos are
//! rejected rather than silently ignored, matching the spec's declarative
//! "atomic cluster update" requirement.

mod dispatch;
mod model;
mod sizes;
mod validate;

pub use dispatch::DispatchRule;
pub use model::{
    Manifest, ModelRef, ModelSpec, Priority, ResourceBudget, RouterSpec, RoutingSpec,
    SpecialistSpec,
};
pub use validate::ManifestValidation;

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read manifest from {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("manifest is invalid: {0}")]
    Invalid(String),
    #[error("size parse failed ({input}): {reason}")]
    BadSize { input: String, reason: String },
}

pub type Result<T> = std::result::Result<T, ManifestError>;

pub fn load_from_str(yaml: &str) -> Result<Manifest> {
    let raw: model::RawManifest = serde_yaml::from_str(yaml)?;
    raw.into_manifest()
}

pub fn load_from_path(path: impl AsRef<Path>) -> Result<Manifest> {
    let p = path.as_ref();
    let yaml = std::fs::read_to_string(p).map_err(|e| ManifestError::Read {
        path: p.display().to_string(),
        source: e,
    })?;
    load_from_str(&yaml)
}
