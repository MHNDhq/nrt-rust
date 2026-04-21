use crate::{dispatch::DispatchRule, sizes::parse_mb, validate::ManifestValidation, ManifestError};
use nrt_core::{ModelId, Tier};
use serde::{Deserialize, Serialize};

/// Fully-validated Manifest. The Cluster Manager consumes this form, never the raw YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub cluster: String,
    pub resources: ResourceBudget,
    pub models: ModelSpec,
    #[serde(default)]
    pub fallback: Option<ModelRef>,
    pub routing: RoutingSpec,
}

impl Manifest {
    /// Every ModelRef the cluster needs to track, including the router and fallback.
    pub fn all_models(&self) -> Vec<&ModelRef> {
        let mut out = Vec::with_capacity(2 + self.models.specialists.len());
        out.push(&self.models.router);
        for s in &self.models.specialists {
            out.push(s);
        }
        if let Some(fb) = &self.fallback {
            out.push(fb);
        }
        out
    }

    pub fn find(&self, id: &ModelId) -> Option<&ModelRef> {
        self.all_models().into_iter().find(|m| &m.id == id)
    }

    /// Lightweight validation (structural). Budget/VRAM math is cluster-side.
    pub fn validate(&self) -> Result<ManifestValidation, ManifestError> {
        ManifestValidation::check(self)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub gpu_budget_mb: u64,
    pub ram_budget_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub router: ModelRef,
    #[serde(default)]
    pub specialists: Vec<ModelRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub id: ModelId,
    pub source: String,
    pub tier: Tier,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub max_active_sessions: Option<u32>,
    #[serde(default)]
    pub co_activation: Option<ModelId>,
    #[serde(default)]
    pub promotion_hint: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Critical,
    High,
    #[default]
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingSpec {
    pub entry: ModelId,
    pub dispatch_rule: DispatchRule,
}

//
// Raw types used for deserialization only. We keep these private because the
// Manifest example uses an asymmetric `models:` layout — `router:` is a
// singleton sub-map while `specialists:` is a list — and a separate raw type
// keeps that asymmetry out of the public API.
//

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawManifest {
    pub cluster: String,
    pub resources: RawResources,
    pub models: RawModelSpec,
    #[serde(default)]
    pub fallback: Option<RawModelRef>,
    pub routing: RawRouting,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawResources {
    pub gpu_budget: String,
    pub ram_budget: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawModelSpec {
    pub router: RouterSpec,
    #[serde(default)]
    pub specialists: Vec<SpecialistSpec>,
}

/// The router entry in the Manifest has no explicit `id` — its id is implicit
/// ("router"). We capture it as a RouterSpec and assign the id post-parse.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterSpec {
    pub source: String,
    pub tier: Tier,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub max_active_sessions: Option<u32>,
    #[serde(default)]
    pub co_activation: Option<ModelId>,
    #[serde(default)]
    pub promotion_hint: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialistSpec {
    pub id: ModelId,
    pub source: String,
    pub tier: Tier,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub max_active_sessions: Option<u32>,
    #[serde(default)]
    pub co_activation: Option<ModelId>,
    #[serde(default)]
    pub promotion_hint: Option<String>,
}

/// Fallback has no explicit id — we assign "fallback" post-parse.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawModelRef {
    pub source: String,
    pub tier: Tier,
    #[serde(default)]
    pub priority: Priority,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawRouting {
    pub entry: ModelId,
    pub dispatch_rule: String,
}

impl RawManifest {
    pub(crate) fn into_manifest(self) -> Result<Manifest, ManifestError> {
        let gpu_budget_mb = parse_mb(&self.resources.gpu_budget)?;
        let ram_budget_mb = parse_mb(&self.resources.ram_budget)?;

        let router = ModelRef {
            id: ModelId::new("router"),
            source: self.models.router.source,
            tier: self.models.router.tier,
            priority: self.models.router.priority,
            max_active_sessions: self.models.router.max_active_sessions,
            co_activation: self.models.router.co_activation,
            promotion_hint: self.models.router.promotion_hint,
        };

        let specialists = self
            .models
            .specialists
            .into_iter()
            .map(|s| ModelRef {
                id: s.id,
                source: s.source,
                tier: s.tier,
                priority: s.priority,
                max_active_sessions: s.max_active_sessions,
                co_activation: s.co_activation,
                promotion_hint: s.promotion_hint,
            })
            .collect();

        let fallback = self.fallback.map(|fb| ModelRef {
            id: ModelId::new("fallback"),
            source: fb.source,
            tier: fb.tier,
            priority: fb.priority,
            max_active_sessions: None,
            co_activation: None,
            promotion_hint: None,
        });

        let dispatch_rule =
            DispatchRule::parse(&self.routing.dispatch_rule).map_err(ManifestError::Invalid)?;

        let manifest = Manifest {
            cluster: self.cluster,
            resources: ResourceBudget {
                gpu_budget_mb,
                ram_budget_mb,
            },
            models: ModelSpec {
                router,
                specialists,
            },
            fallback,
            routing: RoutingSpec {
                entry: self.routing.entry,
                dispatch_rule,
            },
        };

        manifest.validate()?;
        Ok(manifest)
    }
}
