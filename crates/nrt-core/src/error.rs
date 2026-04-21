use thiserror::Error;

pub type NrtResult<T> = Result<T, NrtError>;

#[derive(Debug, Error)]
pub enum NrtError {
    #[error("model {0} not found in cluster")]
    ModelNotFound(String),

    #[error("session {0} not found")]
    SessionNotFound(String),

    #[error("cluster is at capacity (resident={resident}, standby={standby}, budget_vram_mb={budget_vram_mb})")]
    ClusterAtCapacity {
        resident: usize,
        standby: usize,
        budget_vram_mb: u64,
    },

    #[error("tier transition rejected: {from:?} -> {to:?} ({reason})")]
    InvalidTransition {
        from: crate::tier::Tier,
        to: crate::tier::Tier,
        reason: String,
    },

    #[error("dispatch rule could not resolve intent {intent:?} against specialists {known:?}")]
    DispatchUnresolved {
        intent: String,
        known: Vec<String>,
    },

    #[error("backend error: {0}")]
    Backend(String),

    #[error("manifest error: {0}")]
    Manifest(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(String),
}

impl NrtError {
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
    pub fn manifest(msg: impl Into<String>) -> Self {
        Self::Manifest(msg.into())
    }
}
