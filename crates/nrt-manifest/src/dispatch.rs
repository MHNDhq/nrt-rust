//! Parser for the Manifest `dispatch_rule` mini-language.
//!
//! Grammar (v0):
//!
//! ```text
//! rule       := ident '.output.' ident '->' ident '.{' ident '}'
//!            |  ident '.output.' ident '->' ident '.' ident      // literal target
//!            |  'literal:' ident '=' ident (',' ident '=' ident)*
//! ```
//!
//! Examples:
//!   `router.output.intent -> specialists.{intent}`
//!   `router.output.intent -> specialists.billing`
//!
//! The grammar is deliberately small. It captures the 90% case shown in the NRT
//! spec's Manifest example and escapes to application code for anything more
//! exotic. Richer dispatch (conditional, fan-out, regex) is an explicit escape
//! hatch deferred to a future milestone — see the "Manifest becomes a leaky
//! abstraction" risk in the spec.

use nrt_core::ModelId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchRule {
    /// `router.output.intent -> specialists.{intent}`
    /// At inference time: look up the source model's `intent` field on its
    /// InferenceResponse, then dispatch to the specialist with id == intent.
    IntentDispatch {
        source: ModelId,
        field: String,
        target_set: String,
    },
    /// `router.output.intent -> specialists.billing`
    /// Unconditional dispatch, ignores source field.
    FixedRoute {
        source: ModelId,
        target: ModelId,
    },
}

impl DispatchRule {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let (lhs, rhs) = raw.split_once("->").ok_or_else(|| {
            format!("dispatch rule missing '->': {raw:?}")
        })?;
        let lhs = lhs.trim();
        let rhs = rhs.trim();

        // LHS: <source>.output.<field>
        let lhs_parts: Vec<&str> = lhs.split('.').collect();
        if lhs_parts.len() != 3 || lhs_parts[1] != "output" {
            return Err(format!(
                "dispatch rule LHS must be '<model>.output.<field>', got {lhs:?}"
            ));
        }
        let source = ModelId::new(lhs_parts[0]);
        let field = lhs_parts[2].to_string();

        // RHS: <set>.{<var>} for dispatch, or <set>.<id> for fixed.
        if let Some(rest) = rhs.strip_suffix('}') {
            let (set, var) = rest.split_once(".{").ok_or_else(|| {
                format!("dispatch rule RHS is malformed: {rhs:?}")
            })?;
            // The captured var should equal the LHS field name — gentle warning
            // if not, but we don't hard-fail.
            let var = var.trim();
            if !var.eq_ignore_ascii_case(&field) {
                tracing::warn!(
                    "dispatch rule var {:?} does not match LHS field {:?}; using LHS field at runtime",
                    var,
                    field
                );
            }
            return Ok(Self::IntentDispatch {
                source,
                field,
                target_set: set.trim().to_string(),
            });
        }

        // Fixed route: <set>.<target_id>, where we drop the set prefix.
        let rhs_parts: Vec<&str> = rhs.split('.').collect();
        if rhs_parts.len() != 2 {
            return Err(format!(
                "dispatch rule RHS must be '<set>.<id>' or '<set>.{{<var>}}', got {rhs:?}"
            ));
        }
        Ok(Self::FixedRoute {
            source,
            target: ModelId::new(rhs_parts[1]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_intent_dispatch() {
        let r = DispatchRule::parse("router.output.intent -> specialists.{intent}").unwrap();
        match r {
            DispatchRule::IntentDispatch { source, field, target_set } => {
                assert_eq!(source.as_str(), "router");
                assert_eq!(field, "intent");
                assert_eq!(target_set, "specialists");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parses_fixed_route() {
        let r = DispatchRule::parse("router.output.intent -> specialists.billing").unwrap();
        match r {
            DispatchRule::FixedRoute { source, target } => {
                assert_eq!(source.as_str(), "router");
                assert_eq!(target.as_str(), "billing");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn rejects_missing_arrow() {
        assert!(DispatchRule::parse("router.output.intent specialists.{intent}").is_err());
    }

    #[test]
    fn rejects_malformed_lhs() {
        assert!(DispatchRule::parse("router.intent -> specialists.{intent}").is_err());
    }
}
