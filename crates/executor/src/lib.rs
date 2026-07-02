//! The Engine — the two-level control model in one place. The developer switch
//! sets the box mode (top level); below it the autonomous machinery drains the
//! flat work-unit set: admit (homogeneity guard) -> walk the sealed cell-DAG
//! against a protected oracle -> reduce to a unit-verdict -> emit a VerdictEvent
//! -> escalate unit-atomically. No machine-rate process flips the box.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use spark_execution::{consequence_of, reduce_verdict, walk_cells, Oracle, RunCommand, RunState};
use spark_host::{probe_ready, HostCommand, HostHandle, HostSpec, HostState, ResidencyHost, ServingHostView};
use spark_exploration::{SessionCommand, SessionState};
use spark_interface::{
    Artifact, ArtifactBody, ArtifactDelivery, Capabilities, CapabilityManifest, Cell,
    DeliveryCapability, DeliveryResult, Integration, IntegrationMethod, ManifestBinding, Operational,
    Verdict, VerdictEvent, WorkUnit, content_hash,
};
use spark_queue::{decide_admission, HeterogeneityRateView, PrioritySetView, UnitCommand, UnitEvent, UnitState};
use spark_switch::{BoxCommand, BoxState, BoxStatusView, Mode};

/// The top ladder rung: a unit may escalate light(0) -> heavy(1); a rejecting
/// verdict at the heavy binding halts the unit (the ladder is exhausted).
pub const MAX_LADDER: u32 = 1;

/// This executor's stable identity, published in its CapabilityManifest.
pub const EXECUTOR_ID: &str = "dgx-spark-gb10";

/// The reference-substrate CapabilityManifest: what this box can run, published
/// out of band so a producer can match a unit's requirements before dispatch. On
/// the Spark the `bindings` are exactly the resident/validated bindings (see
/// `bindings/qwen3.6-35b-gb10.yaml`): the FP8 MoE day binding served by vLLM.
/// `operational` is filled in at publish time (advisory only — never matched).
pub fn default_manifest() -> CapabilityManifest {
    CapabilityManifest {
        executor_id: EXECUTOR_ID.into(),
        emitted_at: String::new(),
        capabilities: Capabilities {
            bindings: vec![ManifestBinding {
                provider: "local-vllm".into(),
                model_id: "qwen3.6-35b".into(),
                revision: None,
                quantization: "FP8".into(),
            }],
            delivery: DeliveryCapability {
                modes: vec!["inline".into(), "repository".into()],
                url_schemes: vec!["file".into(), "https".into(), "ssh".into()],
                integration_methods: vec!["push-branch".into(), "pull-request".into()],
                forges: vec!["github.com".into()],
            },
            // Shape languages this executor can validate output against. "prose" is
            // the loosest (any text); the structured ones gate machine-checkable output.
            shape_languages: vec!["prose".into(), "JSON".into(), "JSON Schema".into(), "SHACL".into()],
            // Gate kinds it can run against a protected oracle (open vocabulary).
            gate_kinds: vec!["command".into()],
        },
        operational: None,
    }
}

/// A permissive manifest for tests: covers the synthetic `coder` bindings and
/// both delivery modes over `file://`, so the drain tests admit their units. The
/// not-admitted path is exercised separately against the reference manifest.
#[cfg(test)]
fn covering_manifest() -> CapabilityManifest {
    CapabilityManifest {
        executor_id: "test-box".into(),
        emitted_at: "t0".into(),
        capabilities: Capabilities {
            bindings: vec![
                ManifestBinding { provider: String::new(), model_id: "coder".into(), revision: None, quantization: "q4".into() },
                ManifestBinding { provider: String::new(), model_id: "coder".into(), revision: None, quantization: "q8".into() },
            ],
            delivery: DeliveryCapability {
                modes: vec!["inline".into(), "repository".into()],
                url_schemes: vec!["file".into()],
                integration_methods: vec!["push-branch".into(), "pull-request".into()],
                forges: vec![],
            },
            shape_languages: vec![],
            gate_kinds: vec![],
        },
        operational: None,
    }
}

/// A QUEUE-mode engine whose manifest covers the synthetic test bindings.
#[cfg(test)]
fn test_engine() -> Engine {
    let mut e = Engine::new();
    e.manifest = covering_manifest();
    e
}

/// `utilization-view` projector: is the box earning its keep?
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtilizationView {
    pub mode: Mode,
    pub in_flight: i64,
    pub exploring: bool,
    pub discoveries: i64,
}

impl Default for UtilizationView {
    fn default() -> Self {
        UtilizationView { mode: Mode::Off, in_flight: 0, exploring: false, discoveries: 0 }
    }
}

/// The outcome of draining one unit, for reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum DrainOutcome {
    Accepted { unit_ref: String },
    Escalated { unit_ref: String, to_ladder: u32 },
    Halted { unit_ref: String },
    /// The unit NEVER RAN: its derived requirements exceed this box's published
    /// capabilities. Answered with a `not-admitted` verdict carrying the distance.
    NotAdmitted { unit_ref: String, missing: Vec<String> },
    Idle,
    NotQueueMode,
}

/// The full executor state — persistable, so the CLI can drive it across calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engine {
    pub mode: Mode,
    /// The flat, edgeless priority set (front = highest priority).
    pub priority_set: Vec<WorkUnit>,
    pub stream: Vec<VerdictEvent>,
    pub priority_view: PrioritySetView,
    pub heterogeneity_view: HeterogeneityRateView,
    pub verdict_view: VerdictStreamCount,
    pub utilization: UtilizationView,
    /// The physical serving host that materializes the current residency (a vLLM
    /// container on the box). Defaulted so older state.json files still load.
    #[serde(default)]
    pub host: HostState,
    #[serde(default)]
    pub host_handle: Option<HostHandle>,
    #[serde(default)]
    pub serving_host_view: ServingHostView,
    /// This box's published self-description — what can run here. Admission at the
    /// boundary matches a unit's derived requirements against it (authoritatively);
    /// producers match against the published copy before dispatch. Defaulted so
    /// older state.json files still load.
    #[serde(default = "default_manifest")]
    pub manifest: CapabilityManifest,
    seq: u64,
}

/// A serde-friendly mirror of the verdict-stream projector.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerdictStreamCount {
    pub emitted: i64,
}

impl Default for Engine {
    /// A fresh box: OFF, empty queue, and the reference-substrate manifest as its
    /// published capabilities (so a bare `Engine::default()` admits the units the
    /// resident bindings can serve).
    fn default() -> Self {
        Engine {
            mode: Mode::default(),
            priority_set: Vec::new(),
            stream: Vec::new(),
            priority_view: PrioritySetView::default(),
            heterogeneity_view: HeterogeneityRateView::default(),
            verdict_view: VerdictStreamCount::default(),
            utilization: UtilizationView::default(),
            host: HostState::default(),
            host_handle: None,
            serving_host_view: ServingHostView::default(),
            manifest: default_manifest(),
            seq: 0,
        }
    }
}

// PrioritySetView / HeterogeneityRateView need serde to persist; provide it here
// via shadow structs would be heavy — instead we re-derive their numbers from
// events, so we add Serialize/Deserialize on the source types in their crate.

/// Split an `http://host:port` endpoint into `(host, port)` for the probe.
fn host_port(endpoint: &str) -> (String, u16) {
    let s = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let authority = s.split('/').next().unwrap_or(s);
    match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
        None => (authority.to_string(), 80),
    }
}

/// Poll `host:port` until it accepts a connection or `timeout` elapses.
fn wait_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        if probe_ready(host, port, Duration::from_millis(500)) {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine::default()
    }

    /// Publish this box's CapabilityManifest, stamped `now` and carrying the
    /// current advisory `operational` hints (box mode, queue depth). The normative
    /// `capabilities` are the stored self-description; `operational` is snapshot
    /// live and is NEVER used for matching.
    pub fn publish_manifest(&self, now: &str) -> CapabilityManifest {
        let mut m = self.manifest.clone();
        m.emitted_at = now.to_string();
        m.operational = Some(Operational {
            mode: Some(format!("{:?}", self.mode)),
            queue_depth: Some(self.priority_set.len() as i64),
            capacity: None,
        });
        m
    }

    /// Emit a `not-admitted` VerdictEvent for a unit the box cannot cover: nothing
    /// ran, so `tier-ran` and `cell-results` are absent and `missing-capabilities`
    /// carries the distance. Binds to `halt` (the consumer re-routes or escalates
    /// to a human — a higher tier never adds a missing capability). Returns the
    /// outcome; the caller has already removed the unit from the queue.
    fn emit_not_admitted(&mut self, unit: &WorkUnit, missing: Vec<String>, now: &str) -> DrainOutcome {
        self.seq += 1;
        let event = VerdictEvent {
            event_id: format!("ve-{}-{}", unit.unit_ref, self.seq),
            emitted_at: now.to_string(),
            unit_ref: unit.unit_ref.clone(),
            parent_deliverable: unit.parent_deliverable.clone(),
            bundle_hash: unit.bundle_hash.clone(),
            verdict: Verdict::NotAdmitted,
            missing_capabilities: Some(missing.clone()),
            tier_ran: None,
            cell_results: Vec::new(),
            delivery_result: None,
            next_consequence: consequence_of(Verdict::NotAdmitted),
        };
        self.stream.push(event);
        self.verdict_view.emitted += 1;
        DrainOutcome::NotAdmitted { unit_ref: unit.unit_ref.clone(), missing }
    }

    /// Throw the developer switch (top-level control). Returns the new mode, or
    /// the rejected invariant id on a no-op flip.
    pub fn throw_switch(&mut self, mode: Mode) -> Result<Mode, &'static str> {
        let state = BoxState { mode: self.mode };
        let events = state.decide(&BoxCommand::ThrowSwitch { mode })?;
        let mut view = BoxStatusView { mode: self.mode };
        for e in &events {
            view.apply(e);
        }
        self.mode = view.mode;
        self.utilization.mode = view.mode;
        Ok(self.mode)
    }

    /// Physically materialize the residency for the just-thrown mode: retire any
    /// live host (the single-residency rule made physical), launch the new vLLM
    /// host through the seam, then poll its endpoint and confirm it ready. Every
    /// transition is `serving-host-decider`-gated. Returns Ok(true) when the host
    /// answered before `ready_timeout`, Ok(false) when it launched but stayed
    /// cold (the caller should not dispatch to it yet).
    pub fn launch_residency(
        &mut self,
        host: &dyn ResidencyHost,
        spec: &HostSpec,
        ready_timeout: Duration,
    ) -> std::io::Result<bool> {
        // Free VRAM first: a live host must be retired before another launches.
        self.retire_residency(host)?;

        let events = self
            .host
            .decide(&HostCommand::Launch { containerized: true })
            .map_err(|inv| std::io::Error::new(std::io::ErrorKind::Other, inv))?;
        let handle = host.launch(spec)?;
        for e in &events {
            self.host.evolve(e);
            self.serving_host_view.apply(e);
        }
        self.host_handle = Some(handle.clone());

        // Readiness gate: do not confirm (and so do not let work dispatch) until
        // the host's endpoint actually answers.
        let (h, p) = host_port(&handle.endpoint);
        let ready = wait_ready(&h, p, ready_timeout);
        if ready {
            if let Ok(events) = self.host.decide(&HostCommand::ConfirmReady) {
                for e in &events {
                    self.host.evolve(e);
                    self.serving_host_view.apply(e);
                }
            }
        }
        Ok(ready)
    }

    /// Retire the live host (stop its container, free VRAM). A no-op when nothing
    /// is materialized — `inv-nothing-to-retire` simply rejects and we return Ok.
    pub fn retire_residency(&mut self, host: &dyn ResidencyHost) -> std::io::Result<()> {
        if let Ok(events) = self.host.decide(&HostCommand::Retire) {
            if let Some(h) = &self.host_handle {
                host.retire(h)?;
            }
            for e in &events {
                self.host.evolve(e);
                self.serving_host_view.apply(e);
            }
            self.host_handle = None;
        }
        Ok(())
    }

    /// Receive a frozen WorkUnit across the seam and run the admission guard.
    /// Returns Ok(()) when enqueued, Err(invariant) when rejected.
    pub fn admit(&mut self, unit: WorkUnit) -> Result<(), &'static str> {
        match decide_admission(&unit) {
            Ok(events) => {
                for e in &events {
                    self.priority_view.on_queue_event(e);
                    self.heterogeneity_view.apply(e);
                }
                self.priority_set.push(unit);
                Ok(())
            }
            Err(inv) => {
                self.heterogeneity_view.apply(&UnitEvent::Rejected);
                Err(inv)
            }
        }
    }

    /// Reprioritize: move a queued unit to the front of the flat set.
    pub fn reprioritize_to_front(&mut self, unit_ref: &str) -> Result<(), &'static str> {
        let idx = self.priority_set.iter().position(|u| u.unit_ref == unit_ref);
        // A queued unit is, by construction, in the set; the decider guards the rest.
        let guard = if idx.is_some() { UnitState { state: spark_queue::Lifecycle::Queued, ladder: 0 } } else { UnitState::default() };
        guard.decide(&UnitCommand::Reprioritize)?;
        let i = idx.unwrap();
        let u = self.priority_set.remove(i);
        self.priority_set.insert(0, u);
        Ok(())
    }

    /// Drain one unit: the bottom-level autonomous machinery. Only runs in QUEUE.
    pub fn drain_one(&mut self, oracle: &dyn Oracle, now: &str) -> DrainOutcome {
        if self.mode != Mode::Queue {
            return DrainOutcome::NotQueueMode;
        }
        if self.priority_set.is_empty() {
            return DrainOutcome::Idle;
        }
        let unit = self.priority_set.remove(0);
        self.priority_view.on_admitted();

        // Admission (authoritative, before verification): a unit whose derived
        // requirements this box cannot cover never runs — it is answered with a
        // `not-admitted` verdict naming the concrete distance.
        let missing = self.manifest.missing_for(&unit);
        if !missing.is_empty() {
            return self.emit_not_admitted(&unit, missing, now);
        }
        self.utilization.in_flight += 1;

        // ExecutionRun: admit -> walk -> compute -> emit (each decider-gated).
        let mut run = RunState::default();
        for e in run.decide(&RunCommand::Admit { box_mode: self.mode }).expect("queue mode admits") {
            run.evolve(&e);
        }
        let cell_results = walk_cells(&unit.cell_graph, oracle);
        let verdict = reduce_verdict(&cell_results);
        for e in run.decide(&RunCommand::ComputeVerdict).expect("admitted run computes") {
            run.evolve(&e);
        }
        for e in run.decide(&RunCommand::Emit).expect("verdict reached emits") {
            run.evolve(&e);
        }
        self.seq += 1;
        let tier_ran = if unit.ladder_position == 0 { "light" } else { "heavy" };
        let event = VerdictEvent {
            event_id: format!("ve-{}-{}", unit.unit_ref, self.seq),
            emitted_at: now.to_string(),
            unit_ref: unit.unit_ref.clone(),
            parent_deliverable: unit.parent_deliverable.clone(),
            bundle_hash: unit.bundle_hash.clone(),
            verdict,
            missing_capabilities: None,
            tier_ran: Some(tier_ran.to_string()),
            cell_results,
            delivery_result: None,
            next_consequence: consequence_of(verdict),
        };
        self.stream.push(event);
        self.verdict_view.emitted += 1;
        self.utilization.in_flight -= 1;

        match verdict {
            Verdict::Accepted => DrainOutcome::Accepted { unit_ref: unit.unit_ref },
            Verdict::Rejected | Verdict::Escalate | Verdict::NotAdmitted => self.escalate(unit),
        }
    }

    /// Unit-atomic escalation: re-enqueue the whole unit one binding up if the
    /// ladder has headroom, else halt it in failed.
    fn escalate(&mut self, mut unit: WorkUnit) -> DrainOutcome {
        let guard = UnitState { state: spark_queue::Lifecycle::Queued, ladder: unit.ladder_position };
        match guard.decide(&UnitCommand::Escalate { max_ladder: MAX_LADDER }) {
            Ok(events) => {
                for e in &events {
                    self.priority_view.on_queue_event(e);
                }
                unit.ladder_position += 1;
                unit.tier = "heavy".to_string();
                let to = unit.ladder_position;
                let unit_ref = unit.unit_ref.clone();
                self.priority_set.push(unit); // re-enqueued, whole unit, one binding up
                DrainOutcome::Escalated { unit_ref, to_ladder: to }
            }
            Err(_inv) => DrainOutcome::Halted { unit_ref: unit.unit_ref },
        }
    }

    // --- Exploration (only while the box is in EXPLORER) ---

    pub fn start_exploration(&mut self) -> Result<(), &'static str> {
        SessionState::default().decide(&SessionCommand::Start { box_mode: self.mode })?;
        self.utilization.exploring = true;
        Ok(())
    }

    pub fn produce_discovery_record(&mut self) -> Result<(), &'static str> {
        // A started session is active; the decider guards an unstarted one.
        let state = if self.utilization.exploring { SessionState { phase: spark_exploration::Phase::Active } } else { SessionState::default() };
        state.decide(&SessionCommand::ProduceDiscoveryRecord)?;
        self.utilization.exploring = false;
        self.utilization.discoveries += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_interface::{Cell, ModelBinding};
    use std::collections::BTreeMap;

    struct AllPass;
    impl Oracle for AllPass {
        fn gate(&self, _c: &spark_interface::Cell) -> bool { true }
    }
    struct AllFail;
    impl Oracle for AllFail {
        fn gate(&self, _c: &spark_interface::Cell) -> bool { false }
    }

    fn binding(model: &str, q: &str) -> ModelBinding {
        ModelBinding { model: model.into(), quantization: q.into(), params: BTreeMap::new(), ..Default::default() }
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

    fn unit(unit_ref: &str, cells: Vec<Cell>, b: ModelBinding) -> WorkUnit {
        WorkUnit {
            unit_ref: unit_ref.into(),
            parent_deliverable: "d".into(),
            bundle_hash: "sha256:x".into(),
            spmc_bundle: serde_json::json!({}),
            model_binding: b,
            tier: "light".into(),
            cell_graph: cells,
            ..Default::default()
        }
    }

    fn one_cell_unit(r: &str) -> WorkUnit {
        let b = binding("coder", "q4");
        unit(r, vec![exec_cell("c", b.clone(), &[])], b)
    }

    #[test]
    fn strip_code_fences_extracts_the_block() {
        assert_eq!(strip_code_fences("```rust\nfn a(){}\n```"), "fn a(){}");
        assert_eq!(strip_code_fences("here:\n```\nx\n```\ndone"), "x");
        assert_eq!(strip_code_fences("no fences here"), "no fences here");
    }

    #[test]
    fn cannot_drain_outside_queue_mode() {
        let mut e = Engine::new();
        assert_eq!(e.drain_one(&AllPass, "t0"), DrainOutcome::NotQueueMode);
    }

    #[test]
    fn admit_then_drain_accepts_and_emits() {
        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(one_cell_unit("u1")).unwrap();
        assert_eq!(e.priority_view.queued, 1);
        let out = e.drain_one(&AllPass, "t1");
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u1".into() });
        assert_eq!(e.stream.len(), 1);
        assert_eq!(e.stream[0].verdict, Verdict::Accepted);
        assert_eq!(e.priority_view.queued, 0);
    }

    #[test]
    fn a_unit_whose_binding_the_box_cannot_serve_is_not_admitted() {
        // A fresh engine's manifest is the reference substrate (serves only the
        // resident FP8 binding); a `coder@q4` unit's requirement is uncovered.
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(one_cell_unit("u1")).unwrap(); // structurally valid: enqueues
        let out = e.drain_one(&AllPass, "t1");
        match out {
            DrainOutcome::NotAdmitted { unit_ref, missing } => {
                assert_eq!(unit_ref, "u1");
                assert!(missing.contains(&"binding:/coder@q4".to_string()), "distance: {missing:?}");
            }
            other => panic!("expected NotAdmitted, got {other:?}"),
        }
        // A not-admitted verdict was emitted: nothing ran (no tier, no cells).
        let ev = e.stream.last().unwrap();
        assert_eq!(ev.verdict, Verdict::NotAdmitted);
        assert_eq!(ev.tier_ran, None);
        assert!(ev.cell_results.is_empty());
        assert_eq!(ev.next_consequence, spark_interface::Consequence::Halt);
        assert_eq!(e.utilization.in_flight, 0); // never entered flight
    }

    #[test]
    fn heterogeneous_unit_is_rejected_at_admission() {
        let mut e = Engine::new();
        let mixed = unit(
            "bad",
            vec![Cell { cell_id: "c".into(), binding: binding("coder", "q8"), ..Default::default() }],
            binding("coder", "q4"),
        );
        assert_eq!(e.admit(mixed), Err("inv-binding-homogeneity"));
        assert_eq!(e.heterogeneity_view.rejected, 1);
        assert!(e.priority_set.is_empty());
    }

    #[test]
    fn a_failing_unit_escalates_then_halts_when_the_ladder_is_exhausted() {
        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(one_cell_unit("u1")).unwrap();
        // First drain at light(0): fails -> escalate to heavy(1), re-enqueued.
        assert_eq!(e.drain_one(&AllFail, "t1"), DrainOutcome::Escalated { unit_ref: "u1".into(), to_ladder: 1 });
        assert_eq!(e.priority_set.len(), 1);
        // Second drain at heavy(1): fails -> ladder exhausted -> halt.
        assert_eq!(e.drain_one(&AllFail, "t2"), DrainOutcome::Halted { unit_ref: "u1".into() });
        assert!(e.priority_set.is_empty());
        assert_eq!(e.stream.len(), 2);
        assert_eq!(e.stream[1].tier_ran.as_deref(), Some("heavy"));
    }

    #[test]
    fn the_switch_is_the_only_path_between_modes() {
        let mut e = Engine::new();
        // Exploration is refused in the default/off + queue modes.
        e.throw_switch(Mode::Queue).unwrap();
        assert_eq!(e.start_exploration(), Err("inv-box-must-be-explorer"));
        e.throw_switch(Mode::Explorer).unwrap();
        e.start_exploration().unwrap();
        e.produce_discovery_record().unwrap();
        assert_eq!(e.utilization.discoveries, 1);
        assert!(!e.utilization.exploring);
    }

    #[test]
    fn rethrowing_the_resident_mode_is_a_rejected_noop() {
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        assert_eq!(e.throw_switch(Mode::Queue), Err("inv-distinct-mode"));
    }

    #[test]
    fn materializing_a_residency_launches_confirms_ready_then_retires() {
        use spark_host::{HostPhase, HostSpec, LocalProcessHost, ServingHostView};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let host = LocalProcessHost { base_url: format!("http://127.0.0.1:{port}") };
        let spec = HostSpec {
            host_id: "queue-host".into(),
            model: "coder".into(),
            image: "vllm/vllm-openai:latest".into(),
            ssh_target: String::new(),
            port,
            extra_args: vec![],
        };
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();

        let ready = e.launch_residency(&host, &spec, Duration::from_secs(2)).unwrap();
        assert!(ready, "a listening endpoint should confirm ready");
        assert_eq!(e.host.phase, HostPhase::Ready);
        assert_eq!(e.serving_host_view, ServingHostView { materialized: 1, ready: true });
        assert_eq!(e.host_handle.as_ref().unwrap().model, "coder");

        e.retire_residency(&host).unwrap();
        assert_eq!(e.host.phase, HostPhase::Retired);
        assert_eq!(e.serving_host_view, ServingHostView { materialized: 0, ready: false });
        assert!(e.host_handle.is_none());
    }
}

// ───────────────────── production drain (all seams composed) ─────────────

use spark_sandbox::{CredentialBroker, LeaseCommand, LeaseState, SandboxCommand, SandboxRuntime, SandboxState};
use spark_serving::{BatchCommand, BatchState, Worker};
use spark_stream::DurableLog;

/// Assemble a cell's worker prompt from its frozen inputs: the selected context
/// fragments (C, in `context_refs` order), then the artifacts produced by this
/// cell's prerequisites (so a test-first `impl` cell sees the `test` it must
/// satisfy), then the inline prompt (P). No callback — every input is resolved
/// from the unit or from prior cells in this same run. A legacy unit with no
/// inline prompt falls back to a synthetic label.
fn build_prompt(unit: &WorkUnit, cell: &Cell, produced_content: &std::collections::BTreeMap<String, String>) -> String {
    let mut s = String::new();
    for r in &cell.context_refs {
        if let Some(frag) = unit.context_pool.get(r) {
            s.push_str(&frag.content);
            s.push_str("\n\n");
        }
    }
    for dep in &cell.depends_on {
        if let Some(content) = produced_content.get(dep) {
            s.push_str(&format!("--- artifact produced by prerequisite cell `{dep}` ---\n{content}\n\n"));
        }
    }
    if cell.prompt.trim().is_empty() {
        s.push_str(&format!("unit {} cell {}", unit.unit_ref, cell.cell_id));
    } else {
        s.push_str(&cell.prompt);
    }
    s
}

/// The artifact id and workspace-relative path for a cell's output. Falls back to
/// the cell id when the cell did not declare an `output` (a bare contract cell).
fn artifact_id_and_path(cell: &Cell) -> (String, String) {
    match &cell.output {
        Some(o) => {
            let path = o.path.clone().unwrap_or_else(|| format!("{}.out", cell.cell_id));
            (o.artifact_id.clone(), path)
        }
        None => (cell.cell_id.clone(), format!("{}.out", cell.cell_id)),
    }
}

/// Extract the code from a model response: if it fenced the answer in a ```` ```lang ````
/// block, return the block's body; otherwise return the text unchanged. Keeps
/// artifacts compilable when the model adds fences despite being asked not to.
fn strip_code_fences(s: &str) -> String {
    let t = s.trim();
    if let Some(start) = t.find("```") {
        let after = &t[start + 3..];
        // drop the rest of the opening fence line (the language tag, if any)
        let after = match after.find('\n') {
            Some(i) => &after[i + 1..],
            None => after,
        };
        return match after.find("```") {
            Some(end) => after[..end].trim_end().to_string(),
            None => after.trim_end().to_string(),
        };
    }
    s.to_string()
}

/// A dependency order over the cell-graph (Kahn's algorithm). Cells with unmet or
/// dangling deps are appended last so every cell is still processed exactly once.
fn topo_order(cells: &[Cell]) -> Vec<usize> {
    use std::collections::BTreeMap;
    let idx: BTreeMap<&str, usize> = cells.iter().enumerate().map(|(i, c)| (c.cell_id.as_str(), i)).collect();
    let mut indeg = vec![0usize; cells.len()];
    for (i, c) in cells.iter().enumerate() {
        for d in &c.depends_on {
            if idx.contains_key(d.as_str()) {
                indeg[i] += 1;
            }
        }
    }
    let mut queue: Vec<usize> = (0..cells.len()).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::new();
    let mut qi = 0;
    while qi < queue.len() {
        let i = queue[qi];
        qi += 1;
        order.push(i);
        for (j, c) in cells.iter().enumerate() {
            if c.depends_on.iter().any(|d| idx.get(d.as_str()) == Some(&i)) {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    queue.push(j);
                }
            }
        }
    }
    for i in 0..cells.len() {
        if !order.contains(&i) {
            order.push(i);
        }
    }
    order
}

/// Seed a fresh workspace from a real product checkout so the oracle can build and
/// test the artifacts in a real project tree. A git checkout is cloned (fast, and
/// it excludes build output and untracked cruft); anything else is copied,
/// skipping heavy build/VCS dirs.
fn seed_workspace(seed: &std::path::Path, workspace: &std::path::Path) -> std::io::Result<()> {
    if seed.join(".git").is_dir() {
        let status = std::process::Command::new("git")
            .args(["clone", "--quiet", "--local", "--no-hardlinks"])
            .arg(seed)
            .arg(workspace)
            .status();
        if matches!(status, Ok(s) if s.success()) {
            return Ok(());
        }
        // fall through to a plain copy if git is unavailable or the clone failed
    }
    copy_dir_excluding(seed, workspace, &["target", ".git", "node_modules", "bin", "obj", ".spark"])
}

/// Recursive copy of `src` into `dst`, skipping directory names in `exclude`.
fn copy_dir_excluding(src: &std::path::Path, dst: &std::path::Path, exclude: &[&str]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if exclude.iter().any(|e| std::ffi::OsStr::new(e) == name) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if from.is_dir() {
            copy_dir_excluding(&from, &to, exclude)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Ensure the workspace is a git repo with a committed baseline, so a subsequent
/// `git diff` captures exactly what the unit produced. A cloned checkout already
/// has history; a copied/empty tree gets an `init` + baseline commit. Best-effort:
/// if git is unavailable, patch emission simply yields nothing.
fn ensure_git_baseline(workspace: &std::path::Path) {
    if workspace.join(".git").is_dir() {
        return;
    }
    let git = |args: &[&str]| {
        let _ = std::process::Command::new("git").arg("-C").arg(workspace).args(args).status();
    };
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&["-c", "user.email=spark@local", "-c", "user.name=spark", "commit", "-q", "-m", "seed baseline", "--allow-empty"]);
}

/// Emit a reviewable unified diff of everything the unit changed in `workspace` to
/// `out_path`. Returns the path when a non-empty patch was written. Best-effort:
/// no git, or no changes, yields `None`.
fn emit_patch(workspace: &std::path::Path, out_path: &std::path::Path) -> Option<String> {
    if !workspace.join(".git").is_dir() {
        return None;
    }
    // Exclude build output the oracle produced (e.g. `cargo test` creates target/)
    // so the delivered diff is just the unit's source changes.
    let excludes = [":!target", ":!node_modules", ":!bin", ":!obj", ":!.spark", ":!.git"];
    let mut add = std::process::Command::new("git");
    add.arg("-C").arg(workspace).args(["add", "-A", "--", "."]).args(excludes);
    let _ = add.status();
    let mut diff = std::process::Command::new("git");
    diff.arg("-C").arg(workspace).args(["diff", "--cached", "--", "."]).args(excludes);
    let out = diff.output().ok()?;
    if out.stdout.is_empty() {
        return None;
    }
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(out_path, &out.stdout).ok()?;
    Some(out_path.display().to_string())
}

/// Seed a `repository`-delivery working tree by cloning the DECLARED repository
/// into `workspace` and checking out `base_ref`. `file:///…` covers local
/// development; a remote URL covers production (its host is a declared network
/// destination, and push/PR credentials are exchanged executor-side at the
/// boundary — never read from the frozen unit). Concurrency isolation against one
/// repository is the sandbox's job (a per-run clone here), invisible to the seam.
fn clone_repository(url: &str, base_ref: &str, workspace: &std::path::Path) -> std::io::Result<()> {
    // `git clone` refuses a non-empty target; the sandbox provisioned a fresh dir.
    let _ = std::fs::remove_dir_all(workspace);
    let status = std::process::Command::new("git")
        .args(["clone", "--quiet"])
        .arg(url)
        .arg(workspace)
        .status()?;
    if !status.success() {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("git clone {url} failed")));
    }
    if !base_ref.trim().is_empty() {
        let _ = std::process::Command::new("git")
            .arg("-C").arg(workspace)
            .args(["checkout", "--quiet", base_ref])
            .status();
    }
    Ok(())
}

/// Land the produced tree in the declared repository per its integration method,
/// returning the `delivery-result` (branch, commit, `pr-url`). The commit SHA IS
/// the content hash over the produced tree (git canonicalizes it), so provenance
/// is preserved with no artifact payload crossing the seam. Push is best-effort:
/// a `file:///` local repo accepts the branch with no credential; a remote push
/// that fails still yields the local branch + commit as the provenance anchor.
/// For `pull-request` delivery against a known forge, a compare URL is derived.
fn deliver_to_repository(
    workspace: &std::path::Path,
    url: &str,
    integration: &Integration,
    unit_ref: &str,
) -> Option<DeliveryResult> {
    if !workspace.join(".git").is_dir() {
        return None;
    }
    let safe: String = unit_ref.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '/' { c } else { '_' }).collect();
    let branch = if integration.branch_name.trim().is_empty() {
        format!("spark/{safe}")
    } else {
        integration.branch_name.clone()
    };
    let git = |args: &[&str]| {
        std::process::Command::new("git").arg("-C").arg(workspace).args(args).status()
    };
    // A branch for this run, then stage the unit's source changes (excluding build
    // output the oracle produced) and commit them.
    let _ = git(&["checkout", "-q", "-B", &branch]);
    let excludes = [":!target", ":!node_modules", ":!bin", ":!obj", ":!.spark", ":!.git"];
    {
        let mut add = std::process::Command::new("git");
        add.arg("-C").arg(workspace).args(["add", "-A", "--", "."]).args(excludes);
        let _ = add.status();
    }
    let commit_ok = std::process::Command::new("git")
        .arg("-C").arg(workspace)
        .args(["-c", "user.email=spark@local", "-c", "user.name=spark", "commit", "-q", "-m"])
        .arg(format!("spark: {unit_ref}"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !commit_ok {
        // Nothing to commit (no changes) — no ref to report back.
        return None;
    }
    let commit = std::process::Command::new("git")
        .arg("-C").arg(workspace)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;

    // Push the branch to the declared repository (the only permitted destination).
    // `origin` is the clone's remote; push HEAD to the named branch.
    let pushed = std::process::Command::new("git")
        .arg("-C").arg(workspace)
        .args(["push", "--quiet", "origin", &format!("HEAD:refs/heads/{branch}")])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    // For pull-request delivery against a known forge, derive the compare URL the
    // producer reconciles against. (Opening the PR is a forge API call under the
    // executor-side credential; the derived URL is the ref the producer expects.)
    let pr_url = if matches!(integration.method, IntegrationMethod::PullRequest) && pushed {
        let (scheme, host) = spark_interface::url_scheme_and_host(url);
        match (scheme.as_str(), host) {
            ("https", Some(h)) if h == "github.com" => {
                let repo = url.trim_end_matches(".git").trim_start_matches("https://github.com/");
                let base = if integration.target_ref.trim().is_empty() { "main" } else { integration.target_ref.as_str() };
                Some(format!("https://github.com/{repo}/compare/{base}...{branch}?expand=1"))
            }
            _ => None,
        }
    } else {
        None
    };

    Some(DeliveryResult { branch, commit, pr_url })
}

impl Engine {
    /// Drain one unit through the full production path: provision a per-unit
    /// sandbox (optionally SEEDED from a real product checkout), broker
    /// credentials, invoke the worker per cell in dependency order (each cell fed
    /// its prerequisites' artifacts + selected context), VERIFY the produced
    /// artifacts against a protected oracle running in the workspace (e.g. `cargo
    /// test` — the gate that makes `accepted` mean "it builds and tests pass"),
    /// reduce to a verdict, append it to the durable log, emit a reviewable patch
    /// on accept, then tear the sandbox down and revoke creds. Every step is gated
    /// by its decider; effects land only inside the sandbox.
    #[allow(clippy::too_many_arguments)]
    pub fn drain_one_isolated(
        &mut self,
        sandbox: &dyn SandboxRuntime,
        broker: &dyn CredentialBroker,
        worker: &dyn Worker,
        oracle: &dyn Oracle,
        log: &mut DurableLog,
        seed: Option<&std::path::Path>,
        deliveries_dir: Option<&std::path::Path>,
        now: &str,
    ) -> std::io::Result<DrainOutcome> {
        if self.mode != Mode::Queue {
            return Ok(DrainOutcome::NotQueueMode);
        }
        if self.priority_set.is_empty() {
            return Ok(DrainOutcome::Idle);
        }
        let unit = self.priority_set.remove(0);
        self.priority_view.on_admitted();

        // 0. Admission (authoritative, before verification): a unit whose derived
        //    requirements this box cannot cover NEVER RUNS — no sandbox, no worker
        //    call — it is answered with a `not-admitted` verdict naming the distance.
        let missing = self.manifest.missing_for(&unit);
        if !missing.is_empty() {
            let outcome = self.emit_not_admitted(&unit, missing, now);
            if let DrainOutcome::NotAdmitted { .. } = &outcome {
                log.append(self.stream.last().expect("not-admitted event just pushed"))?;
            }
            return Ok(outcome);
        }
        self.utilization.in_flight += 1;

        // The declared repository, when this unit lands in `repository` mode.
        let repository = match &unit.artifact_delivery {
            ArtifactDelivery::Repository { url, base_ref, integration } => Some((url.clone(), base_ref.clone(), integration.clone())),
            ArtifactDelivery::Inline => None,
        };

        // 1. Environment: provision the per-unit sandbox (network declared by
        //    construction). The writable working tree is a per-run CLONE of the
        //    declared repository under `repository` delivery (isolation against one
        //    repo is the sandbox's job, invisible to the seam), else a scratch
        //    workspace optionally SEEDED from a product checkout so the oracle can
        //    build/test the artifacts in a real project tree.
        let mut sbx = SandboxState::default();
        for e in sbx.decide(&SandboxCommand::Provision { network_declared: true }).expect("declared network provisions") {
            sbx.evolve(&e);
        }
        let workspace = sandbox.provision(&unit.unit_ref)?;
        if let Some((url, base_ref, _)) = &repository {
            clone_repository(url, base_ref, &workspace)?;
        } else if let Some(seed) = seed {
            seed_workspace(seed, &workspace)?;
        }
        if deliveries_dir.is_some() && repository.is_none() {
            // A git baseline lets us emit a reviewable diff regardless of how the
            // workspace was seeded (a repository clone already has history).
            ensure_git_baseline(&workspace);
        }

        // 2. Credentials: exchange the grant-reference for a short-lived lease (if any).
        let mut lease = LeaseState::default();
        let token = unit.credential_grant.as_ref().map(|grant| {
            for e in lease.decide(&LeaseCommand::Exchange { sandbox_active: true }).expect("live sandbox leases") {
                lease.evolve(&e);
            }
            broker.exchange(grant)
        });

        // 3. Serving: invoke the worker per cell in DEPENDENCY ORDER, feeding each
        //    cell its selected context (C) and its prerequisites' produced
        //    artifacts (so a test-first `impl` cell sees the `test` it must
        //    satisfy). Each artifact is content-hashed and written INTO the
        //    workspace; its verdict transport follows the declared delivery mode.
        if !unit.cell_graph.is_empty() {
            let mut batch = BatchState::default();
            for e in batch.decide(&BatchCommand::Form { homogeneous: true, nonempty: true }).expect("homogeneous nonempty batch") {
                batch.evolve(&e);
            }
            for e in batch.decide(&BatchCommand::Dispatch).expect("formed batch dispatches") {
                batch.evolve(&e);
            }
        }
        // Inline delivery returns each artifact by value; `repository` delivery
        // carries NO per-cell payloads (the run's landing is reported once, in
        // `delivery-result`). Content is always written into the workspace and
        // tracked so a dependent cell can see its prerequisites' output.
        let mut produced: std::collections::BTreeMap<String, Artifact> = std::collections::BTreeMap::new();
        let mut produced_content: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
        for i in topo_order(&unit.cell_graph) {
            let cell = &unit.cell_graph[i];
            let prompt = build_prompt(&unit, cell, &produced_content);
            if let Ok(raw) = worker.invoke(&cell.binding, &prompt) {
                // Models fence code in ```lang blocks; the artifact is the code
                // itself, so extract it before writing/hashing/delivering.
                let content = strip_code_fences(&raw);
                let (artifact_id, rel_path) = artifact_id_and_path(cell);
                if let Some(parent) = std::path::Path::new(&rel_path).parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = std::fs::create_dir_all(workspace.join(parent));
                    }
                }
                let _ = std::fs::write(workspace.join(&rel_path), &content);
                // Only `inline` delivery emits an artifact body on the verdict.
                if repository.is_none() {
                    produced.insert(
                        cell.cell_id.clone(),
                        Artifact {
                            artifact_id,
                            content_hash: content_hash(content.as_bytes()),
                            delivery: ArtifactBody::Inline { content: content.clone() },
                        },
                    );
                }
                produced_content.insert(cell.cell_id.clone(), content);
            }
        }

        // 4. Verification: run the protected check against the artifacts now in the
        //    workspace. This is the DECISIVE gate — a unit is accepted only if every
        //    cell produced content AND the oracle's command passes. Each cell-result
        //    carries the produced artifact (inline mode) and the shared evidence.
        let report = oracle.verify(&workspace);
        let all_produced = produced_content.len() == unit.cell_graph.len();
        let passed = report.passed && all_produced;
        let mut cell_results: Vec<spark_interface::CellResult> = Vec::new();
        for cell in &unit.cell_graph {
            let produced_here = produced_content.contains_key(&cell.cell_id);
            let mut r = spark_interface::CellResult::gated(cell.cell_id.clone(), passed && produced_here);
            r.artifact = produced.get(&cell.cell_id).cloned();
            r.evidence = report.evidence.clone();
            cell_results.push(r);
        }
        let verdict = reduce_verdict(&cell_results);

        // 4b. Delivery (repository mode): on accept, land the produced tree in the
        //     declared repository per its integration method and capture the
        //     `delivery-result` (branch, commit, `pr-url`) BEFORE teardown. The
        //     commit SHA is the provenance anchor; no artifact payload crosses.
        let delivery_result = match (&repository, verdict) {
            (Some((url, _, integration)), Verdict::Accepted) => {
                deliver_to_repository(&workspace, url, integration, &unit.unit_ref)
            }
            _ => None,
        };

        // 5. Output Contract: emit the VerdictEvent to the durable, idempotent log.
        self.seq += 1;
        let tier_ran = if unit.ladder_position == 0 { "light" } else { "heavy" };
        let event = VerdictEvent {
            event_id: format!("ve-{}-{}", unit.unit_ref, self.seq),
            emitted_at: now.to_string(),
            unit_ref: unit.unit_ref.clone(),
            parent_deliverable: unit.parent_deliverable.clone(),
            bundle_hash: unit.bundle_hash.clone(),
            verdict,
            missing_capabilities: None,
            tier_ran: Some(tier_ran.to_string()),
            cell_results,
            delivery_result,
            next_consequence: consequence_of(verdict),
        };
        log.append(&event)?; // idempotent by bundle_hash
        self.stream.push(event);
        self.verdict_view.emitted += 1;

        // 5b. Delivery (inline mode): on accept, emit a reviewable patch of
        //     everything the unit changed in the workspace, BEFORE teardown
        //     destroys it. This is how code leaves the box under `inline`
        //     delivery — a diff the developer applies. Under `repository`
        //     delivery the run already landed as a pushed branch (delivery-result).
        if verdict == Verdict::Accepted && repository.is_none() {
            if let Some(dir) = deliveries_dir {
                let safe: String = unit.unit_ref.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
                emit_patch(&workspace, &dir.join(format!("{safe}.patch")));
            }
        }

        // 6. Teardown: destroy the sandbox and revoke the lease — nothing standing survives.
        if let Some(t) = &token {
            for e in lease.decide(&LeaseCommand::Revoke).expect("active lease revokes") {
                lease.evolve(&e);
            }
            broker.revoke(t);
        }
        sandbox.teardown(&workspace)?;
        for e in sbx.decide(&SandboxCommand::Teardown).expect("provisioned sandbox tears down") {
            sbx.evolve(&e);
        }
        self.utilization.in_flight -= 1;

        Ok(match verdict {
            Verdict::Accepted => DrainOutcome::Accepted { unit_ref: unit.unit_ref },
            Verdict::Rejected | Verdict::Escalate | Verdict::NotAdmitted => self.escalate(unit),
        })
    }
}

#[cfg(test)]
mod production_tests {
    use super::*;
    use spark_interface::{Cell, Environment, ModelBinding};
    use spark_sandbox::{LocalBroker, LocalSandbox};
    use spark_serving::StubWorker;
    use std::collections::BTreeMap;

    struct PassOracle;
    impl Oracle for PassOracle {
        fn gate(&self, _c: &Cell) -> bool { true }
    }

    fn exec_cell(id: &str, b: ModelBinding, deps: &[&str]) -> Cell {
        Cell {
            cell_id: id.into(),
            binding: b,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            prompt: format!("write the {id} for the widget"),
            schema: serde_json::json!({ "type": "string" }),
            ..Default::default()
        }
    }

    fn unit(r: &str) -> WorkUnit {
        let b = ModelBinding { model: "coder".into(), quantization: "q4".into(), params: BTreeMap::new(), ..Default::default() };
        WorkUnit {
            unit_ref: r.into(),
            parent_deliverable: "d".into(),
            bundle_hash: format!("sha256:{r}"),
            spmc_bundle: serde_json::json!({}),
            model_binding: b.clone(),
            tier: "light".into(),
            cell_graph: vec![
                exec_cell("test", b.clone(), &[]),
                exec_cell("impl", b, &["test"]),
            ],
            environment: Environment { network: vec![], workspace: "ws".into() },
            credential_grant: Some("grant-ref-1".into()),
            ..Default::default()
        }
    }

    #[test]
    fn full_isolated_drain_provisions_runs_gates_emits_and_tears_down() {
        let dir = std::env::temp_dir().join("spark-prod-test");
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("verdicts.jsonl")).unwrap();

        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(unit("u1")).unwrap();

        let out = e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &PassOracle, &mut log, None, None, "t1").unwrap();
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u1".into() });
        assert_eq!(log.len(), 1);                       // durably logged
        assert_eq!(e.utilization.in_flight, 0);         // sandbox torn down
        assert!(!sandbox.root.join("u1").exists());     // workspace destroyed

        // Idempotency: re-emitting the same bundle_hash does not double the log.
        assert!(!log.append(&e.stream[0]).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn real_oracle_failing_verification_escalates_with_evidence() {
        use spark_execution::CommandOracle;
        let dir = std::env::temp_dir().join("spark-oracle-fail");
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("v.jsonl")).unwrap();
        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(unit("u1")).unwrap();

        // A protected oracle whose command fails: the produced artifacts don't pass.
        let oracle = CommandOracle { command: "exit 1".into(), worker_writable: false };
        let out = e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &oracle, &mut log, None, None, "t1").unwrap();
        assert_eq!(out, DrainOutcome::Escalated { unit_ref: "u1".into(), to_ladder: 1 });
        let ev = &e.stream[0];
        assert_eq!(ev.verdict, Verdict::Escalate);
        // Evidence records the failing command + exit code for audit.
        let evidence = ev.cell_results[0].evidence.as_ref().expect("verification evidence present");
        assert_eq!(evidence["exit"], serde_json::json!(1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn real_oracle_passing_accepts_and_delivers_a_patch() {
        use spark_execution::CommandOracle;
        let dir = std::env::temp_dir().join("spark-oracle-pass");
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("v.jsonl")).unwrap();
        let deliveries = dir.join("deliveries");
        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(unit("u1")).unwrap();

        let oracle = CommandOracle { command: "true".into(), worker_writable: false };
        let out = e
            .drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &oracle, &mut log, None, Some(deliveries.as_path()), "t1")
            .unwrap();
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u1".into() });
        // An accepted unit delivers a reviewable diff of what it wrote.
        let patch = deliveries.join("u1.patch");
        assert!(patch.exists(), "accepted unit should deliver a reviewable patch (requires git)");
        let body = std::fs::read_to_string(&patch).unwrap();
        assert!(body.contains("test.out") && body.contains("impl.out"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inline_delivery_returns_content_hashed_artifacts_in_the_verdict() {
        use spark_interface::{ArtifactBody, CellVerdict};
        let dir = std::env::temp_dir().join("spark-inline-test");
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("verdicts.jsonl")).unwrap();

        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(unit("u1")).unwrap(); // default artifact_delivery == inline

        e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &PassOracle, &mut log, None, None, "t1").unwrap();
        let ev = &e.stream[0];
        assert!(ev.delivery_result.is_none(), "inline delivery carries no delivery-result");
        // Every cell accepted and carries an inline, content-hashed artifact.
        for r in &ev.cell_results {
            assert_eq!(r.verdict, CellVerdict::Accepted);
            let a = r.artifact.as_ref().expect("inline delivery returns the artifact by value");
            assert!(a.content_hash.starts_with("sha256:"));
            assert!(matches!(a.delivery, ArtifactBody::Inline { .. }));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A repository-delivery unit that lands its run in a declared `file:///` git
    /// repo: the executor clones it, the worker writes, the oracle passes, and the
    /// run is pushed as a branch — the verdict carries `delivery-result` (branch +
    /// commit) and NO inline artifact bodies.
    #[test]
    fn repository_delivery_pushes_a_branch_and_reports_refs() {
        use spark_interface::{ArtifactDelivery, Integration, IntegrationMethod};
        // A `git` binary is required; skip cleanly if absent (this is a unit test).
        let git_ok = std::process::Command::new("git").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
        if !git_ok {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join("spark-repo-delivery");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Stand up a bare-ish origin repo with an initial commit on `main`.
        let origin = dir.join("origin");
        std::fs::create_dir_all(&origin).unwrap();
        let git = |args: &[&str], cwd: &std::path::Path| {
            std::process::Command::new("git").arg("-C").arg(cwd).args(args).status().unwrap();
        };
        git(&["init", "-q", "-b", "main"], &origin);
        std::fs::write(origin.join("README.md"), "seed\n").unwrap();
        git(&["add", "-A"], &origin);
        git(&["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], &origin);
        // Allow pushes into this non-bare repo's checked-out branch.
        git(&["config", "receive.denyCurrentBranch", "updateInstead"], &origin);

        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("v.jsonl")).unwrap();
        let mut e = test_engine();
        e.throw_switch(Mode::Queue).unwrap();

        let mut u = unit("u-repo");
        u.artifact_delivery = ArtifactDelivery::Repository {
            url: format!("file://{}", origin.display()),
            base_ref: "main".into(),
            integration: Integration {
                method: IntegrationMethod::PushBranch,
                target_ref: "main".into(),
                branch_name: "spark/u-repo".into(),
            },
        };
        e.admit(u).unwrap();

        let oracle = PassOracle;
        let out = e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &oracle, &mut log, None, None, "t1").unwrap();
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u-repo".into() });

        let ev = e.stream.last().unwrap();
        let dr = ev.delivery_result.as_ref().expect("repository delivery reports refs");
        assert_eq!(dr.branch, "spark/u-repo");
        assert_eq!(dr.commit.len(), 40, "a full commit SHA");
        assert!(dr.pr_url.is_none(), "push-branch delivery opens no PR");
        // Repository mode carries no per-cell artifact bodies.
        for r in &ev.cell_results {
            assert!(r.artifact.is_none(), "repository delivery carries refs, not payloads");
        }
        // The branch actually landed in the origin.
        let branches = std::process::Command::new("git").arg("-C").arg(&origin).args(["branch", "--list", "spark/u-repo"]).output().unwrap();
        assert!(String::from_utf8_lossy(&branches.stdout).contains("spark/u-repo"), "branch pushed to origin");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn isolated_drain_outside_queue_is_refused() {
        let dir = std::env::temp_dir().join("spark-prod-test2");
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("v.jsonl")).unwrap();
        let mut e = Engine::new();
        assert_eq!(
            e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &PassOracle, &mut log, None, None, "t0").unwrap(),
            DrainOutcome::NotQueueMode
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
