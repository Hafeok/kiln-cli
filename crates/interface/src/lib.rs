//! The Work-Unit Interface — the seam between the specification pillar
//! (product-cli) and this executor. Three data shapes cross it: a `WorkUnit`
//! travels *in* by value (its SPMC bundle inlined, identified by `bundle_hash`),
//! a `VerdictEvent` travels *out* by event (self-describing, fire-and-forget),
//! and a [`CapabilityManifest`] is *published* out of band — the one seam artifact
//! the execution side authors, matched against a unit's requirements before
//! dispatch. No shared runtime surface; either side is rebuildable against this
//! contract.
//!
//! The **wire** is the canonical [AI Development Contracts](https://github.com/Hafeok/ai-development-contracts)
//! encoding (checked against contracts `0.2.0`, the repository-delivery +
//! CapabilityManifest revision): the kebab-case JSON of
//! [`work-unit.schema.json`](https://github.com/Hafeok/ai-development-contracts/blob/main/schemas/work-unit.schema.json)
//! in, and the kebab-case [`verdict-event.schema.json`](https://github.com/Hafeok/ai-development-contracts/blob/main/schemas/verdict-event.schema.json)
//! out. The structs in this module are kiln's **internal, flattened** projection
//! — a WorkUnit admitted from the wire is parsed as the canonical nested shape
//! ([`canonical::CanonicalWorkUnit`]) and mapped in via [`WorkUnit::from_canonical_json`].
//! Keeping the internal model separate is the contract's rule made concrete: your
//! internal representation is yours; you map to/from a normative encoding *at the
//! seam*, and validate against that encoding's schema. kiln reads the
//! Execution-Contract additions (`environment`, `credential_grant`, `tool_grants`)
//! when a producer carries them as a documented extension, and defaults them to a
//! conformant floor when absent; it validates every incoming unit against the
//! structural invariants before admission.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The pinned SPMC Model axis for a whole unit. It is a *binding*, not a name:
/// identity (provider, model, optional revision) + served quantization +
/// invocation params. On the Spark quantization is load-bearing (it is what makes
/// a model fit 128GB), so two units at the same model name but different
/// quantization are different bindings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBinding {
    /// The serving provider/endpoint identity (contract `binding.provider`), part
    /// of the capability identifier a CapabilityManifest matches on.
    #[serde(default)]
    pub provider: String,
    pub model: String,
    /// The pinned revision, when the producer declared one (contract
    /// `binding.revision`). Carried for attribution; does not enter the match key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub quantization: String,
    #[serde(default)]
    pub params: BTreeMap<String, String>,
}

impl ModelBinding {
    /// The identity two cells must share to be binding-homogeneous.
    pub fn key(&self) -> String {
        let params: Vec<String> = self.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{}/{}@{}[{}]", self.provider, self.model, self.quantization, params.join(","))
    }

    /// The prefixed capability identifier a CapabilityManifest matches this binding
    /// against, by string equality: `binding:<provider>/<model-id>@<quantization>`
    /// (the contract-defined convention; invocation params are not part of it — a
    /// box that serves the binding serves it at any sampling).
    pub fn capability_id(&self) -> String {
        format!("binding:{}/{}@{}", self.provider, self.model, self.quantization)
    }
}

/// The artifact a cell produces — its declared Output-Contract shape (S). The
/// `artifact_id` is echoed on the VerdictEvent so a consumer can match the
/// produced artifact to the cell that declared it. `path` is the tree-relative
/// location the artifact lands at under `repository` delivery.
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
    /// The shape *language* of `schema` (contract `schema.shape-language`, e.g.
    /// "JSON Schema", "SHACL", "prose"). A cell's requirement `shape-language:<name>`
    /// derives from this; an executor admits the unit only if its manifest lists it.
    #[serde(default)]
    pub shape_language: String,
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

/// How a produced run lands inside a declared git repository (contract
/// `artifact-delivery.repository.integration`). `push-branch` pushes a branch to
/// the repository; `pull-request` additionally opens a PR against `target-ref`
/// (a declared capability — a forge API call). The producer must know this to
/// reconcile the ref shape that comes back in `delivery-result`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntegrationMethod {
    PushBranch,
    PullRequest,
}

impl Default for IntegrationMethod {
    fn default() -> Self {
        IntegrationMethod::PushBranch
    }
}

impl IntegrationMethod {
    /// The prefixed capability identifier: `integration:push-branch` etc.
    pub fn capability_id(&self) -> &'static str {
        match self {
            IntegrationMethod::PushBranch => "integration:push-branch",
            IntegrationMethod::PullRequest => "integration:pull-request",
        }
    }
}

/// The integration a repository delivery declares: the method, the ref it targets
/// (the PR base, or the branch pushed against), and the branch name to produce.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Integration {
    pub method: IntegrationMethod,
    #[serde(default, rename = "target-ref")]
    pub target_ref: String,
    #[serde(default, rename = "branch-name")]
    pub branch_name: String,
}

/// The declared transport for produced work (Output Contract: destination).
/// `inline` returns artifact bodies by value in the VerdictEvent; `repository`
/// lands the run in a **declared git repository** (`file:///` for local
/// development, remote for production) per its integration method — and ONLY
/// there. The seam forbids any undeclared side channel; no credential material
/// rides here (repository access is exchanged executor-side from the grant).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum ArtifactDelivery {
    Inline,
    Repository {
        /// The declared repository. `file:///…` local, or a remote (`https`, `ssh`).
        url: String,
        /// The ref the working tree is based on (defaults to the repo's default branch).
        #[serde(default, rename = "base-ref")]
        base_ref: String,
        integration: Integration,
    },
}

impl Default for ArtifactDelivery {
    fn default() -> Self {
        ArtifactDelivery::Inline
    }
}

/// Split a repository URL into its `url-scheme` and (for remote schemes) the forge
/// host, for capability-requirement derivation. `file:///…` → (`file`, None);
/// `https://host/…` → (`https`, Some(host)); `ssh://…` or `git@host:…` →
/// (`ssh`, Some(host)).
pub fn url_scheme_and_host(url: &str) -> (String, Option<String>) {
    if let Some(rest) = url.strip_prefix("file://") {
        let _ = rest;
        return ("file".into(), None);
    }
    for scheme in ["https", "http", "ssh", "git"] {
        if let Some(rest) = url.strip_prefix(&format!("{scheme}://")) {
            let host = rest.split(['/', ':']).next().filter(|h| !h.is_empty()).map(String::from);
            let s = if scheme == "http" { "https" } else { scheme };
            return (s.into(), host);
        }
    }
    // scp-like `git@host:owner/repo.git`
    if let Some((userhost, _)) = url.split_once(':') {
        if let Some((_, host)) = userhost.split_once('@') {
            return ("ssh".into(), Some(host.to_string()));
        }
    }
    ("file".into(), None)
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

    /// The unit's **requirements** as prefixed capability identifiers, derived from
    /// the unit *itself* — no extra declaration. An executor is able to run this
    /// unit iff its CapabilityManifest covers every identifier here (empty distance).
    ///
    /// - `binding:<provider>/<model-id>@<quantization>` — the one pinned binding.
    /// - `delivery:<mode>` and, for `repository`, `url-scheme:<scheme>`,
    ///   `integration:<method>`, and `forge:<host>` for a remote repository.
    /// - `shape-language:<name>` — every distinct cell shape language.
    /// - `gate:<kind>` — every distinct cell gate kind (`gate.kind`).
    pub fn requirements(&self) -> BTreeSet<String> {
        let mut req = BTreeSet::new();
        req.insert(self.model_binding.capability_id());
        match &self.artifact_delivery {
            ArtifactDelivery::Inline => {
                req.insert("delivery:inline".into());
            }
            ArtifactDelivery::Repository { url, integration, .. } => {
                req.insert("delivery:repository".into());
                let (scheme, host) = url_scheme_and_host(url);
                req.insert(format!("url-scheme:{scheme}"));
                req.insert(integration.method.capability_id().to_string());
                if let Some(h) = host {
                    req.insert(format!("forge:{h}"));
                }
            }
        }
        for c in &self.cell_graph {
            if !c.shape_language.trim().is_empty() {
                req.insert(format!("shape-language:{}", c.shape_language));
            }
            if let Some(gate) = &c.gate {
                if let Some(kind) = gate.get("kind").and_then(|k| k.as_str()) {
                    req.insert(format!("gate:{kind}"));
                }
            }
        }
        req
    }
}

/// The Execution Contract verdict vocabulary — a declared artifact, never a bool.
/// `not-admitted` is the contract-tier extension: the unit NEVER EXECUTED because
/// the box lacked a required capability, so it carries `missing-capabilities` and
/// omits `tier-ran` / `cell-results`. Admission failure ≠ execution failure — a
/// higher tier fixes the latter and can never add a missing capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Accepted,
    Rejected,
    Escalate,
    NotAdmitted,
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

/// The produced-artifact body as it travels in the VerdictEvent. Only `inline`
/// delivery returns bodies by value; `repository` delivery carries **no per-cell
/// payloads** — the run's landing is reported once, in `delivery-result` (refs,
/// not bodies), and the commit SHA is the content hash over the produced tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum ArtifactBody {
    Inline { content: String },
}

/// A produced artifact returned by value under `inline` delivery, content-hash
/// identified. Under `repository` delivery no `Artifact` is emitted per cell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Artifact {
    /// Echoes the cell's declared `output.artifact_id`.
    pub artifact_id: String,
    pub content_hash: String,
    pub delivery: ArtifactBody,
}

/// Where a `repository`-delivery run landed (contract `delivery-result`): the
/// branch pushed, the commit SHA (which IS the content hash over the produced
/// tree — git provides the canonicalization), and, for `pull-request` delivery,
/// the opened PR URL. Refs, never payloads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct DeliveryResult {
    pub branch: String,
    pub commit: String,
    #[serde(default, rename = "pr-url", skip_serializing_if = "Option::is_none")]
    pub pr_url: Option<String>,
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
/// kebab-case contract encoding on the wire. `tier_ran` and `cell_results` are
/// present when the unit executed and absent on `not-admitted` (nothing ran);
/// `missing_capabilities` is the inverse — required with `not-admitted`, absent
/// otherwise. `delivery_result` is present only under `repository` delivery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct VerdictEvent {
    pub event_id: String,
    pub emitted_at: String,
    pub unit_ref: String,
    pub parent_deliverable: String,
    pub bundle_hash: String,
    pub verdict: Verdict,
    /// The DISTANCE, required with `not-admitted` and absent otherwise: the
    /// prefixed capability identifiers the box lacks for this unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing_capabilities: Option<Vec<String>>,
    /// The tier the unit actually executed at. Absent on `not-admitted`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_ran: Option<String>,
    /// Per-cell evidence from walking the sealed cell-DAG. Absent on `not-admitted`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cell_results: Vec<CellResult>,
    /// Where a `repository`-delivery run landed (branch, commit, `pr-url`). Absent
    /// under `inline` delivery (artifact bodies ride on `cell-results` instead).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_result: Option<DeliveryResult>,
    pub next_consequence: Consequence,
}

/// Schema 3 — the **CapabilityManifest** the executor publishes out of band (not
/// per run): its declared answer to "what can run here," so a producer can compute
/// a unit's executability at planning time and a deploy target is selectable as
/// "a box whose manifest covers this unit's requirements." This is the one seam
/// artifact the execution side *authors* (the complement of producer-owns).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct CapabilityManifest {
    /// Stable identity of this executor instance (e.g. a box id).
    pub executor_id: String,
    /// When this snapshot was published. Manifests go stale; a match is pre-flight.
    pub emitted_at: String,
    /// NORMATIVE for admission matching — the stable facts.
    pub capabilities: Capabilities,
    /// ADVISORY ONLY — routing hints, never admission facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operational: Option<Operational>,
}

/// The normative capability facts a producer matches a unit's requirements against.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Capabilities {
    /// Model bindings this box can serve — on the Spark, exactly the resident/
    /// validated bindings, pinned with the same identity fields a WorkUnit carries.
    pub bindings: Vec<ManifestBinding>,
    pub delivery: DeliveryCapability,
    /// Shape languages the executor can validate output against.
    pub shape_languages: Vec<String>,
    /// Gate kinds the executor can run (open vocabulary; matched by string equality).
    pub gate_kinds: Vec<String>,
}

/// A servable binding as the manifest declares it (contract identity fields).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ManifestBinding {
    pub provider: String,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub quantization: String,
}

impl ManifestBinding {
    /// `binding:<provider>/<model-id>@<quantization>` — the matched identifier.
    pub fn capability_id(&self) -> String {
        format!("binding:{}/{}@{}", self.provider, self.model_id, self.quantization)
    }
}

/// The delivery modes the box supports and, for `repository`, its reachable
/// schemes / methods / forges.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct DeliveryCapability {
    /// `inline` | `repository`.
    pub modes: Vec<String>,
    #[serde(default)]
    pub url_schemes: Vec<String>,
    #[serde(default)]
    pub integration_methods: Vec<String>,
    #[serde(default)]
    pub forges: Vec<String>,
}

/// Advisory routing hints — MUST NOT make a unit inexecutable.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Operational {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_depth: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<i64>,
}

impl CapabilityManifest {
    /// The full set of prefixed capability identifiers this manifest covers —
    /// what a unit's `requirements()` are matched (by string equality) against.
    /// Derived from `capabilities` only; `operational` is advisory and excluded.
    pub fn capability_ids(&self) -> BTreeSet<String> {
        let mut ids = BTreeSet::new();
        for b in &self.capabilities.bindings {
            ids.insert(b.capability_id());
        }
        for m in &self.capabilities.delivery.modes {
            ids.insert(format!("delivery:{m}"));
        }
        for s in &self.capabilities.delivery.url_schemes {
            ids.insert(format!("url-scheme:{s}"));
        }
        for m in &self.capabilities.delivery.integration_methods {
            ids.insert(format!("integration:{m}"));
        }
        for f in &self.capabilities.delivery.forges {
            ids.insert(format!("forge:{f}"));
        }
        for l in &self.capabilities.shape_languages {
            ids.insert(format!("shape-language:{l}"));
        }
        for k in &self.capabilities.gate_kinds {
            ids.insert(format!("gate:{k}"));
        }
        ids
    }

    /// The **distance** for a unit: `requirements(unit) − capabilities(box)`, as a
    /// sorted vec of the prefixed identifiers the box lacks. Empty distance ⇒
    /// executable (a match). Pre-flight only — the boundary stays authoritative.
    pub fn missing_for(&self, unit: &WorkUnit) -> Vec<String> {
        let have = self.capability_ids();
        unit.requirements().into_iter().filter(|r| !have.contains(r)).collect()
    }

    /// Whether the manifest covers every requirement of the unit (empty distance).
    pub fn covers(&self, unit: &WorkUnit) -> bool {
        self.missing_for(unit).is_empty()
    }
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

/// A JSON value rendered as the string kiln's flat model carries: a bare string
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
/// (ai-development-contracts v0.2.0). kiln admits *this* shape and maps it into
/// the flattened internal [`WorkUnit`] via `From`. These are deserialize-only:
/// the wire is canonical, kiln's interior is its own concern.
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
        // kiln reads them when a producer carries them as an extension, else floors them.
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
        /// Prompt text or a structured message array — kiln carries it as a string.
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
    /// Flatten the canonical contract shape into kiln's internal model. The
    /// unit-level `spmc-bundle.model.binding` becomes both the unit binding and
    /// every cell's binding (the contract has no per-cell binding — tier
    /// homogeneity is a unit property), so `is_binding_homogeneous()` holds by
    /// construction. `context-pool.fragments[]` becomes the id→fragment map;
    /// `cell-graph.cells[]` becomes the cell vec; `requires`→`depends_on`,
    /// `schema.document`→`schema`, `prompt.content`→`prompt`.
    fn from(c: canonical::CanonicalWorkUnit) -> Self {
        let b = &c.spmc_bundle.model.binding;
        let binding = ModelBinding {
            provider: b.provider.clone(),
            model: b.model_id.clone(),
            revision: b.revision.clone(),
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
                shape_language: cell.schema.shape_language,
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
    /// canonical, and the returned `WorkUnit` is kiln's internal projection. The
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
        ModelBinding {
            provider: "local-vllm".into(),
            model: model.into(),
            revision: None,
            quantization: q.into(),
            params: BTreeMap::new(),
        }
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

        let repo = ArtifactDelivery::Repository {
            url: "file:///repo".into(),
            base_ref: "main".into(),
            integration: Integration {
                method: IntegrationMethod::PushBranch,
                target_ref: "main".into(),
                branch_name: "kiln/wu-1".into(),
            },
        };
        let j = serde_json::to_value(&repo).unwrap();
        assert_eq!(j["mode"], "repository");
        assert_eq!(j["url"], "file:///repo");
        assert_eq!(j["base-ref"], "main");
        assert_eq!(j["integration"]["method"], "push-branch");
        assert_eq!(j["integration"]["branch-name"], "kiln/wu-1");
        assert_eq!(serde_json::from_value::<ArtifactDelivery>(j).unwrap(), repo);
    }

    #[test]
    fn url_scheme_and_host_classifies_repositories() {
        assert_eq!(url_scheme_and_host("file:///srv/repo.git"), ("file".into(), None));
        assert_eq!(
            url_scheme_and_host("https://github.com/acme/widget.git"),
            ("https".into(), Some("github.com".into()))
        );
        assert_eq!(
            url_scheme_and_host("ssh://git@gitlab.example/acme/widget.git"),
            ("ssh".into(), Some("git@gitlab.example".into()))
        );
        assert_eq!(
            url_scheme_and_host("git@github.com:acme/widget.git"),
            ("ssh".into(), Some("github.com".into()))
        );
    }

    #[test]
    fn requirements_derive_from_the_unit_itself() {
        let b = binding("coder", "fp8");
        let mut cell = exec_cell("a", b.clone(), &[]);
        cell.shape_language = "JSON Schema".into();
        cell.gate = Some(serde_json::json!({ "kind": "command" }));
        let mut u = unit_with(vec![cell], b);
        u.artifact_delivery = ArtifactDelivery::Repository {
            url: "https://github.com/acme/widget.git".into(),
            base_ref: "main".into(),
            integration: Integration {
                method: IntegrationMethod::PullRequest,
                target_ref: "main".into(),
                branch_name: "kiln/a".into(),
            },
        };
        let req = u.requirements();
        assert!(req.contains("binding:local-vllm/coder@fp8"));
        assert!(req.contains("delivery:repository"));
        assert!(req.contains("url-scheme:https"));
        assert!(req.contains("integration:pull-request"));
        assert!(req.contains("forge:github.com"));
        assert!(req.contains("shape-language:JSON Schema"));
        assert!(req.contains("gate:command"));
    }

    #[test]
    fn manifest_covers_a_unit_or_names_the_distance() {
        let b = binding("coder", "fp8");
        let u = unit_with(vec![exec_cell("a", b.clone(), &[])], b); // inline, no shape/gate
        // A manifest that serves the binding + inline delivery covers it.
        let m = CapabilityManifest {
            executor_id: "box-1".into(),
            emitted_at: "t0".into(),
            capabilities: Capabilities {
                bindings: vec![ManifestBinding {
                    provider: "local-vllm".into(),
                    model_id: "coder".into(),
                    revision: None,
                    quantization: "fp8".into(),
                }],
                delivery: DeliveryCapability { modes: vec!["inline".into()], ..Default::default() },
                shape_languages: vec![],
                gate_kinds: vec![],
            },
            operational: None,
        };
        assert!(m.covers(&u), "distance was {:?}", m.missing_for(&u));

        // Drop the binding → the unit is not admissible; the distance names it.
        let mut m2 = m.clone();
        m2.capabilities.bindings.clear();
        assert_eq!(m2.missing_for(&u), vec!["binding:local-vllm/coder@fp8".to_string()]);
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
            missing_capabilities: None,
            tier_ran: Some("constrained-implementer".into()),
            cell_results: vec![CellResult::gated("c-test", true)],
            delivery_result: None,
            next_consequence: Consequence::Advance,
        };
        let j = serde_json::to_value(&ev).unwrap();
        for k in ["event-id", "emitted-at", "unit-ref", "parent-deliverable", "bundle-hash", "tier-ran", "cell-results", "next-consequence"] {
            assert!(j.get(k).is_some(), "missing kebab-case key {k}: {j}");
        }
        assert_eq!(j["verdict"], "accepted");
        assert_eq!(j["cell-results"][0]["cell-id"], "c-test");
        // absent fields are omitted, not null
        assert!(j.get("missing-capabilities").is_none());
        assert!(j.get("delivery-result").is_none());
        // round-trips back through the same struct
        assert_eq!(serde_json::from_value::<VerdictEvent>(j).unwrap(), ev);
    }

    #[test]
    fn not_admitted_verdict_omits_run_fields_and_names_the_distance() {
        let ev = VerdictEvent {
            event_id: "ev-2".into(),
            emitted_at: "2026-07-02T00:00:00Z".into(),
            unit_ref: "wu-2".into(),
            parent_deliverable: "dl-1".into(),
            bundle_hash: "sha256:def".into(),
            verdict: Verdict::NotAdmitted,
            missing_capabilities: Some(vec!["binding:local-vllm/exotic@nvfp4".into()]),
            tier_ran: None,
            cell_results: vec![],
            delivery_result: None,
            next_consequence: Consequence::Halt,
        };
        let j = serde_json::to_value(&ev).unwrap();
        assert_eq!(j["verdict"], "not-admitted");
        assert_eq!(j["missing-capabilities"][0], "binding:local-vllm/exotic@nvfp4");
        // nothing ran: tier-ran and cell-results are absent
        assert!(j.get("tier-ran").is_none());
        assert!(j.get("cell-results").is_none());
        assert_eq!(serde_json::from_value::<VerdictEvent>(j).unwrap(), ev);
    }
}
