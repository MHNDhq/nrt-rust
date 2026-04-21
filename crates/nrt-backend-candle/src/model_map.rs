use crate::ModelProfile;
use nrt_core::ModelId;
use std::collections::HashMap;

/// Immutable-by-cloning registry of which ModelId maps to which HuggingFace
/// GGUF profile. Populated at backend construction; read-mostly at runtime.
#[derive(Debug, Default, Clone)]
pub struct ModelMap {
    inner: HashMap<ModelId, ModelProfile>,
}

impl ModelMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: ModelId, profile: ModelProfile) {
        self.inner.insert(id, profile);
    }

    pub fn with(mut self, id: ModelId, profile: ModelProfile) -> Self {
        self.inner.insert(id, profile);
        self
    }

    pub fn get(&self, id: &ModelId) -> Option<&ModelProfile> {
        self.inner.get(id)
    }

    pub fn ids(&self) -> Vec<ModelId> {
        self.inner.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
