//! The NRT Cluster Manager.
//!
//! Owns:
//!   - Registry: ModelId -> loaded handle + live tier
//!   - Session registry: SessionId -> Session with optional KV handle
//!   - LRU eviction policy over idle sessions
//!   - Co-activation warming driven by Manifest hints
//!   - Dispatch: router output -> specialist invocation
//!
//! Does NOT own:
//!   - Real weight loading (that's the Backend trait)
//!   - HTTP surface (that's nrt-server)
//!   - Per-session linear allocator (that's a future milestone inside a real backend)
//!
//! The Cluster Manager is the "brain" — it knows the workload shape via the
//! Manifest and knows the resource shape via live telemetry from the Backend.
//! Every other crate is either giving it information or taking its decisions.

pub mod dispatch;
pub mod manager;
pub mod metrics;
pub mod scheduler;

pub use manager::{ClusterHandle, ClusterManager, ClusterSnapshot, CompletionResult};
pub use scheduler::{LruPolicy, SchedulerEvent};
