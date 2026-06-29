//! The Work-Unit Interface — the seam between the specification pillar
//! (product-cli) and this executor. Two data shapes cross it: a `WorkUnit`
//! travels *in* by value (its SPMC bundle inlined, identified by `bundle_hash`),
//! and a `VerdictEvent` travels *out* by event (self-describing, fire-and-forget).
//! No shared runtime surface; either side is rebuildable against this contract.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The pinned SPMC Model axis for a whole unit. It is a *binding*, not a name:
/// identity + served quantization + invocation params. On the Spark quantization
/// is load-bearing (it is what makes a model fit 128GB), so two units at the same
/// model name but different quantization are different bindings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBinding {
    pub model: String,
    pub quantization: String,
    #[serde(default)]
    pub params: BTreeMap<String, String>,
}

impl ModelBinding {
    /// The identity two cells must share to be binding-homogeneous.
    pub fn key(&self) -> String {
        let params: Vec<String> = self.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{}@{}[{}]", self.model, self.quantization, params.join(","))
    }
}

/// A cell — the execution primitive beneath a unit's bundle, living entirely
/// inside the sealed cell-graph. Carries only intra-unit dependencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub cell_id: String,
    pub binding: ModelBinding,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// Whether a green verdict may auto-commit (a *consumer* decision, not the
/// executor's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceptanceClass {
    AutoCommitIfGreen,
    NeedsVerdict,
}

/// The declared execution boundary for a unit (Execution Contract: Environment).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    #[serde(default)]
    pub network: Vec<String>,
    pub workspace: String,
}

/// A cell-invokable tool, classified by whether it may fire mid-execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolClass {
    /// May fire during execution (file write, service call) — a leaf effect.
    Effect,
    /// May NOT fire mid-run; all knowledge was frozen at dispatch.
    Knowledge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolGrant {
    pub name: String,
    pub class: ToolClass,
}

/// The frozen, by-value WorkUnit the executor consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkUnit {
    pub unit_ref: String,
    pub parent_deliverable: String,
    pub bundle_hash: String,
    pub spmc_bundle: serde_json::Value,
    pub model_binding: ModelBinding,
    pub tier: String,
    pub acceptance_class: AcceptanceClass,
    #[serde(default)]
    pub ladder_position: u32,
    #[serde(default)]
    pub cell_graph: Vec<Cell>,
    #[serde(default)]
    pub environment: Environment,
    #[serde(default)]
    pub credential_grant: Option<String>,
    #[serde(default)]
    pub tool_grants: Vec<ToolGrant>,
}

impl WorkUnit {
    /// The binding-homogeneity invariant: a unit is well-formed iff every cell
    /// requires the same model binding as the unit.
    pub fn is_binding_homogeneous(&self) -> bool {
        let unit = self.model_binding.key();
        self.cell_graph.iter().all(|c| c.binding.key() == unit)
    }
}

/// The Execution Contract verdict vocabulary — a declared artifact, never a bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Accepted,
    Rejected,
    Escalate,
}

/// The Transition Contract binding: verdict -> consequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Consequence {
    Advance,
    Halt,
    Retry,
    Escalate,
}

/// Per-cell evidence for the unit verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellResult {
    pub cell_id: String,
    pub passed: bool,
}

/// The self-describing VerdictEvent emitted to the stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerdictEvent {
    pub event_id: String,
    pub emitted_at: String,
    pub unit_ref: String,
    pub parent_deliverable: String,
    pub bundle_hash: String,
    pub verdict: Verdict,
    pub tier_ran: String,
    #[serde(default)]
    pub cell_results: Vec<CellResult>,
    pub next_consequence: Consequence,
}

/// Content hash of a frozen bundle — the unit's identity, and the provenance
/// link a verdict carries. Stable: serialized through `serde_json` which orders
/// object keys deterministically for `BTreeMap`-backed values.
pub fn bundle_hash(bundle: &serde_json::Value) -> String {
    let canonical = serde_json::to_vec(bundle).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&canonical);
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(model: &str, q: &str) -> ModelBinding {
        ModelBinding { model: model.into(), quantization: q.into(), params: BTreeMap::new() }
    }

    fn unit_with(cells: Vec<Cell>, b: ModelBinding) -> WorkUnit {
        WorkUnit {
            unit_ref: "u1".into(),
            parent_deliverable: "d1".into(),
            bundle_hash: "sha256:x".into(),
            spmc_bundle: serde_json::json!({}),
            model_binding: b,
            tier: "light".into(),
            acceptance_class: AcceptanceClass::AutoCommitIfGreen,
            ladder_position: 0,
            cell_graph: cells,
            environment: Environment::default(),
            credential_grant: None,
            tool_grants: vec![],
        }
    }

    #[test]
    fn homogeneous_unit_passes() {
        let b = binding("coder", "q4_K_M");
        let u = unit_with(
            vec![
                Cell { cell_id: "a".into(), binding: b.clone(), depends_on: vec![] },
                Cell { cell_id: "b".into(), binding: b.clone(), depends_on: vec!["a".into()] },
            ],
            b,
        );
        assert!(u.is_binding_homogeneous());
    }

    #[test]
    fn quantization_difference_breaks_homogeneity() {
        // Same model NAME, different served quantization => different binding.
        let u = unit_with(
            vec![Cell { cell_id: "a".into(), binding: binding("coder", "q8_0"), depends_on: vec![] }],
            binding("coder", "q4_K_M"),
        );
        assert!(!u.is_binding_homogeneous());
    }

    #[test]
    fn bundle_hash_is_stable_and_prefixed() {
        let h = bundle_hash(&serde_json::json!({"a": 1, "b": 2}));
        assert!(h.starts_with("sha256:"));
        assert_eq!(h, bundle_hash(&serde_json::json!({"a": 1, "b": 2})));
    }
}
