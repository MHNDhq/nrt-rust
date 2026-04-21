//! Shared types for the Neurometric Runtime prototype.
//!
//! This crate owns the vocabulary every other crate shares: tiers, session IDs,
//! the backend trait. Nothing here depends on YAML parsing, HTTP, or the
//! cluster scheduler — those live upstream and compose these primitives.

pub mod backend;
pub mod error;
pub mod ids;
pub mod kv_cache;
pub mod session;
pub mod tier;

pub use backend::{Backend, BackendLoadHandle, InferenceRequest, InferenceResponse, Token};
pub use error::{NrtError, NrtResult};
pub use ids::{ModelId, SessionId};
pub use kv_cache::{KvCache, KvCacheHandle, KvFootprint};
pub use session::{Session, SessionState};
pub use tier::{Tier, TierTransition};
