use serde::{Deserialize, Serialize};
use std::fmt;

/// Three tiers per NRT spec Pillar 1: weights live in VRAM (Resident), spill
/// to system RAM (Standby), or are served from a remote endpoint (Remote).
/// Active is a session-level qualifier (KV cache allocated for generation),
/// not a weight-residency tier, but it shares the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Weights mapped into VRAM / unified memory. Ready for immediate forward pass.
    Resident,
    /// Weights in system RAM. Promotable to Resident via PCIe or unified-memory zero-copy.
    Standby,
    /// Session has an allocated KV cache and a live generation. Implies Resident weights.
    Active,
    /// Weights live on a different host. Used for fallback/burst.
    Remote,
}

impl Tier {
    pub fn is_local(self) -> bool {
        matches!(self, Self::Resident | Self::Standby | Self::Active)
    }

    pub fn consumes_vram(self) -> bool {
        matches!(self, Self::Resident | Self::Active)
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Resident => "resident",
            Self::Standby => "standby",
            Self::Active => "active",
            Self::Remote => "remote",
        };
        f.write_str(s)
    }
}

/// Record emitted when a model/session moves between tiers.
/// Consumed by tracing spans and the benchmark harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierTransition {
    pub model_id: crate::ModelId,
    pub from: Tier,
    pub to: Tier,
    pub elapsed_ms: u64,
    pub reason: String,
}
