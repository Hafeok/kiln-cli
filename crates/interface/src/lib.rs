//! The Work-Unit Interface — the seam between the specification pillar
//! (product-cli) and this executor. Two data shapes cross it: a `WorkUnit`
//! travels *in* by value (its SPMC bundle inlined, identified by `bundle_hash`),
//! and a `VerdictEvent` travels *out* by event (self-describing, fire-and-forget).
//! No shared runtime surface; either side is rebuildable against this contract.
//!
//! The **wire** is the canonical [AI Development Contracts](https://github.com/Hafeok/ai-development-contracts)
//! encoding (checked against contracts `0.1.0`): the kebab-case JSON of
//! [`work-unit.schema.json`](https://github.com/Hafeok/ai-development-contracts/blob/main/schemas/work-unit.schema.json)
//! in, and the kebab-case [`verdict-event.schema.json`](https://github.com/Hafeok/ai-development-contracts/blob/main/schemas/verdict-event.schema.json)
//! out. The structs in this module are spark's **internal, flattened** projection
//! — a WorkUnit admitted from the wire is parsed as the canonical nested shape
//! ([`canonical::CanonicalWorkUnit`]) and mapped in via [`WorkUnit::from_canonical_json`].
//! Keeping the internal model separate is the contract's rule made concrete: your
//! internal representation is yours; you map to/from a normative encoding *at the
//! seam*, and validate against that encoding's schema. spark reads the
//! Execution-Contract additions (`environment`, `credential_grant`, `tool_grants`)
//! when a producer carries them as a documented extension, and defaults them to a
//! conformant floor when absent; it validates every incoming unit against the
//! structural invariants before admission.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The pinned SPMC Model axis for a whole unit. It is a *binding*, not a name:
/// identity + served quantization + invocation params. On the Spark quantization
/// is load-bearing (it is what makes a model fit 128GB), so two units at the same
/// model name but different quantization are different bindings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

/// The artifact a cell produces — its declared Output-Contract shape (S). The
/// `artifact_id` is echoed on the VerdictEvent so a consumer can match the
/// produced artifact to the cell that declared it. `path` is meaningful under
/// workspace delivery (where the artifact lands under the declared workspace).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CellOutput {
    pub artifact_id: String,
    #[serde(default)]
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// A cell — the execution primitive beneath a unit's bundle, living entirely
/// inside the sealed cell-graph. One cell is one LLM call carrying its own S
/// (`schema`), P (`prompt`) and C (`context_refs`, ids into the unit's context
/// pool). Carries only intra-unit dependencies (`depends_on` = the `requires`
/// edge, never crossing a unit boundary).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub cell_id: String,
    pub binding: ModelBinding,
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// P — the prompt content, inline (never a callback reference).
    #[serde(default)]
    pub prompt: String,
    /// S — the shape document the cell's output must conform to, inline.
    #[serde(default)]
    pub schema: serde_json::Value,
    /// C — fragment ids selected from this unit's `context_pool`.
    #[serde(default)]
    pub context_refs: Vec<String>,
    /// The artifact this cell produces (artifact-id, media-type, path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<CellOutput>,
    /// An optional extra gate beyond schema conformance (interior open).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<serde_json::Value>,
}

/// A content-addressed context fragment (C). Every fragment is inline with a
/// provenance tag; cells select fragments by their id (the map key).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextFragment {
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// Whether a green verdict may auto-commit (a *consumer* decision, not the
/// executor's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceptanceClass {
    AutoCommitIfGreen,
    NeedsVerdict,
}

impl Default for AcceptanceClass {
    fn default() -> Self {
        AcceptanceClass::AutoCommitIfGreen
    }
}

/// The declared transport for produced artifacts (Output Contract: destination).
/// `inline` returns artifacts by value in the VerdictEvent; `workspace` writes
/// them into the declared workspace (a local git clone, a mounted dir) and ONLY
/// there — the seam forbids undeclared side channels either way.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum ArtifactDelivery {
    Inline,
    Workspace {
        #[serde(default)]
        kind: String,
        #[serde(default)]
        location: String,
        #[serde(default, rename = "ref")]
        reference: String,
    },
}

impl Default for ArtifactDelivery {
    fn default() -> Self {
        ArtifactDelivery::Inline
    }
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
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkUnit {
    pub unit_ref: String,
    pub parent_deliverable: String,
    pub bundle_hash: String,
    pub spmc_bundle: serde_json::Value,
    /// M — the one pinned binding for the whole unit (`spmc-bundle.model`).
    pub model_binding: ModelBinding,
    /// A routing handle OVER the binding: which residency to run at. The binding
    /// is what determines output; `tier` picks the box mode / resident model.
    pub tier: String,
    pub acceptance_class: AcceptanceClass,
    #[serde(default)]
    pub ladder_position: u32,
    /// The declared transport for artifacts this unit produces. Defaults to
    /// `inline` — a bare contract WorkUnit returns artifacts by value.
    #[serde(default)]
    pub artifact_delivery: ArtifactDelivery,
    /// C — the content-addressed fragment pool (`spmc-bundle.context-pool`).
    /// Every `context_refs` id on a cell must resolve here.
    #[serde(default)]
    pub context_pool: BTreeMap<String, ContextFragment>,
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

    /// Validate an incoming unit before admission (the consumer-checklist
    /// "validate against the normative schema + the cross-unit-edge structural
    /// check, reject non-conforming units"). Returns the specific invariant id
    /// on the first violation, in a fixed order:
    ///
    /// 1. `inv-binding-homogeneity` — a cell wants a different binding than the unit.
    /// 2. `inv-cross-unit-edge` — a `requires` edge names a cell not in this unit.
    /// 3. `inv-unresolved-context-ref` — a `context_refs` id is not in the pool.
    /// 4. `inv-cell-unexecutable` — a cell is missing its prompt (P) or shape (S).
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.is_binding_homogeneous() {
            return Err("inv-binding-homogeneity");
        }
        let ids: BTreeSet<&str> = self.cell_graph.iter().map(|c| c.cell_id.as_str()).collect();
        for c in &self.cell_graph {
            for d in &c.depends_on {
                if !ids.contains(d.as_str()) {
                    return Err("inv-cross-unit-edge");
                }
            }
            for r in &c.context_refs {
                if !self.context_pool.contains_key(r) {
                    return Err("inv-unresolved-context-ref");
                }
            }
            if c.prompt.trim().is_empty() || c.schema.is_null() {
                return Err("inv-cell-unexecutable");
            }
        }
        Ok(())
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

/// A single cell's verdict from walking the sealed cell-DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CellVerdict {
    Accepted,
    Rejected,
    /// Never gated — a predecessor failed, so this cell was not run.
    Skipped,
}

/// The produced-artifact body as it travels in the VerdictEvent: inline by value,
/// or a reference into the declared workspace. Content-hash lives on `Artifact`
/// so an accepted artifact is attributable in either mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum ArtifactBody {
    Inline {
        content: String,
    },
    Workspace {
        path: String,
        #[serde(rename = "ref")]
        reference: String,
    },
}

/// A produced artifact, content-hash identified in BOTH delivery modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Artifact {
    /// Echoes the cell's declared `output.artifact_id`.
    pub artifact_id: String,
    pub content_hash: String,
    pub delivery: ArtifactBody,
}

fn cell_verdict_default() -> CellVerdict {
    CellVerdict::Skipped
}

/// Per-cell evidence for the unit verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct CellResult {
    pub cell_id: String,
    /// The cell's declared verdict (accepted | rejected | skipped).
    #[serde(default = "cell_verdict_default")]
    pub verdict: CellVerdict,
    /// Convenience predicate the rollup reduces over (== verdict is Accepted).
    pub passed: bool,
    /// The produced artifact, present when the cell ran and produced one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Artifact>,
    /// Gate output / validation report (interior open).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<serde_json::Value>,
}

impl CellResult {
    /// A gated result: accepted when it passed, rejected otherwise.
    pub fn gated(cell_id: impl Into<String>, passed: bool) -> Self {
        CellResult {
            cell_id: cell_id.into(),
            verdict: if passed { CellVerdict::Accepted } else { CellVerdict::Rejected },
            passed,
            artifact: None,
            evidence: None,
        }
    }

    /// A skipped result: a predecessor failed, so the cell was never gated.
    pub fn skipped(cell_id: impl Into<String>) -> Self {
        CellResult {
            cell_id: cell_id.into(),
            verdict: CellVerdict::Skipped,
            passed: false,
            artifact: None,
            evidence: None,
        }
    }
}

/// The self-describing VerdictEvent emitted to the stream — the canonical
/// kebab-case contract encoding on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
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

/// Content hash of arbitrary bytes, `sha256:`-prefixed — used for both the frozen
/// bundle's identity and each produced artifact's `content_hash`.
pub fn content_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

/// Content hash of a frozen bundle — the unit's identity, and the provenance
/// link a verdict carries. Stable: serialized through `serde_json` which orders
/// object keys deterministically for `BTreeMap`-backed values.
pub fn bundle_hash(bundle: &serde_json::Value) -> String {
    let canonical = serde_json::to_vec(bundle).unwrap_or_default();
    content_hash(&canonical)
}

/// A JSON value rendered as the string spark's flat model carries: a bare string
/// passes through; anything structured (a message array, a number) is serialized.
fn json_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// The **canonical contract wire types** — the exact kebab-case shape of
/// [`work-unit.schema.json`](https://github.com/Hafeok/ai-development-contracts/blob/main/schemas/work-unit.schema.json)
/// (ai-development-contracts v0.1.0). spark admits *this* shape and maps it into
/// the flattened internal [`WorkUnit`] via `From`. These are deserialize-only:
/// the wire is canonical, spark's interior is its own concern.
pub mod canonical {
    use super::*;

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalWorkUnit {
        pub unit_ref: String,
        pub parent_deliverable: String,
        pub bundle_hash: String,
        pub tier: String,
        pub acceptance_class: AcceptanceClass,
        #[serde(default)]
        pub ladder_position: u32,
        pub artifact_delivery: ArtifactDelivery,
        pub spmc_bundle: SpmcBundle,
        pub cell_graph: CellGraph,
        // Execution-Contract additions ride OUTSIDE the closed contract envelope;
        // spark reads them when a producer carries them as an extension, else floors them.
        #[serde(default)]
        pub environment: Option<Environment>,
        #[serde(default)]
        pub credential_grant: Option<String>,
        #[serde(default)]
        pub tool_grants: Option<Vec<ToolGrant>>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct SpmcBundle {
        pub model: ModelSection,
        pub context_pool: ContextPool,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct ModelSection {
        #[serde(default)]
        pub capability_tag: Option<String>,
        pub binding: CanonicalBinding,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalBinding {
        pub provider: String,
        pub model_id: String,
        #[serde(default)]
        pub revision: Option<String>,
        #[serde(default)]
        pub architecture: Option<String>,
        pub quantization: String,
        #[serde(default)]
        pub invocation: BTreeMap<String, serde_json::Value>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct ContextPool {
        #[serde(default)]
        pub bundle_form_profile: Option<String>,
        #[serde(default)]
        pub fragments: Vec<CanonicalFragment>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalFragment {
        pub id: String,
        #[serde(default)]
        pub media_type: Option<String>,
        #[serde(default)]
        pub role: Option<String>,
        pub content: serde_json::Value,
        #[serde(default)]
        pub provenance: Option<serde_json::Value>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub struct CellGraph {
        pub cells: Vec<CanonicalCell>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalCell {
        pub id: String,
        #[serde(default)]
        pub requires: Vec<String>,
        pub schema: CanonicalSchema,
        pub prompt: CanonicalPrompt,
        #[serde(default)]
        pub context_refs: Vec<String>,
        pub output: CanonicalOutput,
        #[serde(default)]
        pub gate: Option<serde_json::Value>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalSchema {
        pub shape_language: String,
        #[serde(default)]
        pub shape_version: Option<String>,
        pub document: serde_json::Value,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalPrompt {
        /// Prompt text or a structured message array — spark carries it as a string.
        pub content: serde_json::Value,
        #[serde(default)]
        pub prompt_version: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub struct CanonicalOutput {
        pub artifact_id: String,
        #[serde(default)]
        pub media_type: Option<String>,
        #[serde(default)]
        pub path: Option<String>,
        #[serde(default)]
        pub description: Option<String>,
    }
}

impl From<canonical::CanonicalWorkUnit> for WorkUnit {
    /// Flatten the canonical contract shape into spark's internal model. The
    /// unit-level `spmc-bundle.model.binding` becomes both the unit binding and
    /// every cell's binding (the contract has no per-cell binding — tier
    /// homogeneity is a unit property), so `is_binding_homogeneous()` holds by
    /// construction. `context-pool.fragments[]` becomes the id→fragment map;
    /// `cell-graph.cells[]` becomes the cell vec; `requires`→`depends_on`,
    /// `schema.document`→`schema`, `prompt.content`→`prompt`.
    fn from(c: canonical::CanonicalWorkUnit) -> Self {
        let b = &c.spmc_bundle.model.binding;
        let binding = ModelBinding {
            model: b.model_id.clone(),
            quantization: b.quantization.clone(),
            params: b.invocation.iter().map(|(k, v)| (k.clone(), json_to_string(v))).collect(),
        };

        let context_pool = c
            .spmc_bundle
            .context_pool
            .fragments
            .into_iter()
            .map(|f| {
                let provenance = f
                    .provenance
                    .as_ref()
                    .map(json_to_string)
                    .or(f.role.clone());
                (f.id, ContextFragment { content: json_to_string(&f.content), provenance })
            })
            .collect();

        let cell_graph = c
            .cell_graph
            .cells
            .into_iter()
            .map(|cell| Cell {
                cell_id: cell.id,
                binding: binding.clone(),
                depends_on: cell.requires,
                prompt: json_to_string(&cell.prompt.content),
                schema: cell.schema.document,
                context_refs: cell.context_refs,
                output: Some(CellOutput {
                    artifact_id: cell.output.artifact_id,
                    media_type: cell.output.media_type.unwrap_or_default(),
                    path: cell.output.path,
                }),
                gate: cell.gate,
            })
            .collect();

        let environment = c.environment.unwrap_or_else(|| Environment {
            network: Vec::new(),
            workspace: format!("{}-ws", c.unit_ref),
        });

        WorkUnit {
            unit_ref: c.unit_ref,
            parent_deliverable: c.parent_deliverable,
            bundle_hash: c.bundle_hash,
            spmc_bundle: serde_json::Value::Null, // overwritten with the raw bundle by from_canonical_json
            model_binding: binding,
            tier: c.tier,
            acceptance_class: c.acceptance_class,
            ladder_position: c.ladder_position,
            artifact_delivery: c.artifact_delivery,
            context_pool,
            cell_graph,
            environment,
            credential_grant: c.credential_grant,
            tool_grants: c.tool_grants.unwrap_or_default(),
        }
    }
}

impl WorkUnit {
    /// Admit a WorkUnit from the **canonical contract JSON** (kebab-case,
    /// ai-development-contracts v0.1.0). This is the seam entry point: the wire is
    /// canonical, and the returned `WorkUnit` is spark's internal projection. The
    /// raw `spmc-bundle` is preserved verbatim on `spmc_bundle` so the unit's
    /// identity remains hashable/auditable against what was emitted.
    pub fn from_canonical_json(text: &str) -> serde_json::Result<WorkUnit> {
        let raw: serde_json::Value = serde_json::from_str(text)?;
        let canon: canonical::CanonicalWorkUnit = serde_json::from_value(raw.clone())?;
        let mut unit: WorkUnit = canon.into();
        unit.spmc_bundle = raw.get("spmc-bundle").cloned().unwrap_or(serde_json::Value::Null);
        Ok(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(model: &str, q: &str) -> ModelBinding {
        ModelBinding { model: model.into(), quantization: q.into(), params: BTreeMap::new() }
    }

    fn exec_cell(id: &str, b: ModelBinding, deps: &[&str]) -> Cell {
        Cell {
            cell_id: id.into(),
            binding: b,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            prompt: format!("do {id}"),
            schema: serde_json::json!({ "type": "string" }),
            ..Default::default()
        }
    }

    fn unit_with(cells: Vec<Cell>, b: ModelBinding) -> WorkUnit {
        WorkUnit {
            unit_ref: "u1".into(),
            parent_deliverable: "d1".into(),
            bundle_hash: "sha256:x".into(),
            spmc_bundle: serde_json::json!({}),
            model_binding: b,
            tier: "light".into(),
            cell_graph: cells,
            ..Default::default()
        }
    }

    #[test]
    fn homogeneous_unit_passes() {
        let b = binding("coder", "q4_K_M");
        let u = unit_with(
            vec![exec_cell("a", b.clone(), &[]), exec_cell("b", b.clone(), &["a"])],
            b,
        );
        assert!(u.is_binding_homogeneous());
        assert_eq!(u.validate(), Ok(()));
    }

    #[test]
    fn quantization_difference_breaks_homogeneity() {
        // Same model NAME, different served quantization => different binding.
        let u = unit_with(
            vec![exec_cell("a", binding("coder", "q8_0"), &[])],
            binding("coder", "q4_K_M"),
        );
        assert!(!u.is_binding_homogeneous());
        assert_eq!(u.validate(), Err("inv-binding-homogeneity"));
    }

    #[test]
    fn cross_unit_requires_edge_is_rejected() {
        let b = binding("coder", "q4");
        // cell "a" requires "ghost", which is not a cell in this unit.
        let u = unit_with(vec![exec_cell("a", b.clone(), &["ghost"])], b);
        assert_eq!(u.validate(), Err("inv-cross-unit-edge"));
    }

    #[test]
    fn unresolved_context_ref_is_rejected() {
        let b = binding("coder", "q4");
        let mut cell = exec_cell("a", b.clone(), &[]);
        cell.context_refs = vec!["frag-missing".into()];
        let u = unit_with(vec![cell], b);
        assert_eq!(u.validate(), Err("inv-unresolved-context-ref"));
    }

    #[test]
    fn resolved_context_ref_passes() {
        let b = binding("coder", "q4");
        let mut cell = exec_cell("a", b.clone(), &[]);
        cell.context_refs = vec!["frag-1".into()];
        let mut u = unit_with(vec![cell], b);
        u.context_pool.insert(
            "frag-1".into(),
            ContextFragment { content: "the ADR text".into(), provenance: Some("ADR-071".into()) },
        );
        assert_eq!(u.validate(), Ok(()));
    }

    #[test]
    fn cell_missing_prompt_or_schema_is_unexecutable() {
        let b = binding("coder", "q4");
        let mut cell = exec_cell("a", b.clone(), &[]);
        cell.prompt = "   ".into(); // whitespace-only == absent
        let u = unit_with(vec![cell], b.clone());
        assert_eq!(u.validate(), Err("inv-cell-unexecutable"));

        let mut cell2 = exec_cell("a", b.clone(), &[]);
        cell2.schema = serde_json::Value::Null;
        let u2 = unit_with(vec![cell2], b);
        assert_eq!(u2.validate(), Err("inv-cell-unexecutable"));
    }

    #[test]
    fn bundle_hash_is_stable_and_prefixed() {
        let h = bundle_hash(&serde_json::json!({"a": 1, "b": 2}));
        assert!(h.starts_with("sha256:"));
        assert_eq!(h, bundle_hash(&serde_json::json!({"a": 1, "b": 2})));
    }

    #[test]
    fn content_hash_is_stable() {
        assert_eq!(content_hash(b"hello"), content_hash(b"hello"));
        assert!(content_hash(b"hello").starts_with("sha256:"));
    }

    #[test]
    fn artifact_delivery_serde_round_trips_both_modes() {
        let inline = ArtifactDelivery::Inline;
        let j = serde_json::to_value(&inline).unwrap();
        assert_eq!(j, serde_json::json!({ "mode": "inline" }));
        assert_eq!(serde_json::from_value::<ArtifactDelivery>(j).unwrap(), inline);

        let ws = ArtifactDelivery::Workspace {
            kind: "git".into(),
            location: "/repo".into(),
            reference: "main".into(),
        };
        let j = serde_json::to_value(&ws).unwrap();
        assert_eq!(j["mode"], "workspace");
        assert_eq!(j["ref"], "main");
        assert_eq!(serde_json::from_value::<ArtifactDelivery>(j).unwrap(), ws);
    }

    const CANONICAL_UNIT: &str = r#"{
      "unit-ref": "wu-1",
      "parent-deliverable": "dl-1",
      "bundle-hash": "sha256:abc123",
      "tier": "constrained-implementer",
      "acceptance-class": "needs-verdict",
      "ladder-position": 0,
      "artifact-delivery": { "mode": "inline" },
      "spmc-bundle": {
        "model": {
          "capability-tag": "constrained-implementer",
          "binding": {
            "provider": "example-endpoint",
            "model-id": "coder-m",
            "quantization": "fp8",
            "invocation": { "temperature": 0 }
          }
        },
        "context-pool": {
          "bundle-form-profile": "n-quads-canonical",
          "fragments": [
            { "id": "frag-a", "role": "constraint", "content": "refund_total <= paid_total" }
          ]
        }
      },
      "cell-graph": {
        "cells": [
          {
            "id": "c-test",
            "requires": [],
            "schema": { "shape-language": "JSON Schema", "document": { "type": "object" } },
            "prompt": { "content": "Write failing tests." },
            "context-refs": ["frag-a"],
            "output": { "artifact-id": "art-tests", "media-type": "text/x-rust" }
          },
          {
            "id": "c-impl",
            "requires": ["c-test"],
            "schema": { "shape-language": "JSON Schema", "document": { "type": "object" } },
            "prompt": { "content": "Make them pass." },
            "context-refs": ["frag-a"],
            "output": { "artifact-id": "art-impl", "media-type": "text/x-rust" }
          }
        ]
      }
    }"#;

    #[test]
    fn admits_canonical_contract_json() {
        let u = WorkUnit::from_canonical_json(CANONICAL_UNIT).expect("parses canonical JSON");

        // envelope
        assert_eq!(u.unit_ref, "wu-1");
        assert_eq!(u.parent_deliverable, "dl-1");
        assert_eq!(u.bundle_hash, "sha256:abc123");
        assert_eq!(u.tier, "constrained-implementer");
        assert_eq!(u.acceptance_class, AcceptanceClass::NeedsVerdict);

        // model binding flattened from spmc-bundle.model.binding
        assert_eq!(u.model_binding.model, "coder-m");
        assert_eq!(u.model_binding.quantization, "fp8");
        assert_eq!(u.model_binding.params.get("temperature").map(String::as_str), Some("0"));

        // context-pool.fragments[] -> id-keyed map
        assert!(u.context_pool.contains_key("frag-a"));

        // cell-graph.cells[] -> cells; requires -> depends_on; prompt.content -> prompt
        assert_eq!(u.cell_graph.len(), 2);
        assert_eq!(u.cell_graph[1].cell_id, "c-impl");
        assert_eq!(u.cell_graph[1].depends_on, vec!["c-test".to_string()]);
        assert_eq!(u.cell_graph[0].prompt, "Write failing tests.");
        assert_eq!(u.cell_graph[0].output.as_ref().unwrap().artifact_id, "art-tests");

        // every cell binding == unit binding (homogeneity by construction), and it validates
        assert!(u.is_binding_homogeneous());
        assert_eq!(u.validate(), Ok(()));

        // raw spmc-bundle preserved for identity/audit
        assert!(u.spmc_bundle.get("model").is_some());
    }

    #[test]
    fn verdict_event_serializes_kebab_case() {
        let ev = VerdictEvent {
            event_id: "ev-1".into(),
            emitted_at: "2026-07-02T00:00:00Z".into(),
            unit_ref: "wu-1".into(),
            parent_deliverable: "dl-1".into(),
            bundle_hash: "sha256:abc123".into(),
            verdict: Verdict::Accepted,
            tier_ran: "constrained-implementer".into(),
            cell_results: vec![CellResult::gated("c-test", true)],
            next_consequence: Consequence::Advance,
        };
        let j = serde_json::to_value(&ev).unwrap();
        for k in ["event-id", "emitted-at", "unit-ref", "parent-deliverable", "bundle-hash", "tier-ran", "cell-results", "next-consequence"] {
            assert!(j.get(k).is_some(), "missing kebab-case key {k}: {j}");
        }
        assert_eq!(j["verdict"], "accepted");
        assert_eq!(j["cell-results"][0]["cell-id"], "c-test");
        // round-trips back through the same struct
        assert_eq!(serde_json::from_value::<VerdictEvent>(j).unwrap(), ev);
    }
}
