//! Execution — the executor. Admits a unit (QUEUE-gated) into a per-unit
//! sandbox, walks the sealed cell-DAG gating each cell against a protected
//! oracle, reduces cell-verdicts to a unit-verdict, and emits a VerdictEvent.
//! Realises `execution-run-decider` and `verdict-stream-view`.

use std::path::Path;

use kiln_interface::{Cell, CellResult, Consequence, Verdict};
use kiln_switch::Mode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    New,
    Admitted,
    VerdictReached,
    Emitted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunEvent {
    UnitAdmitted,
    UnitVerdictComputed,
    VerdictEmitted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunCommand {
    Admit { box_mode: Mode },
    ComputeVerdict,
    Emit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    pub phase: Phase,
}

impl Default for RunState {
    fn default() -> Self {
        RunState { phase: Phase::New }
    }
}

impl RunState {
    pub fn evolve(&mut self, event: &RunEvent) {
        match event {
            RunEvent::UnitAdmitted => self.phase = Phase::Admitted,
            RunEvent::UnitVerdictComputed => self.phase = Phase::VerdictReached,
            RunEvent::VerdictEmitted => self.phase = Phase::Emitted,
        }
    }

    /// `execution-run-decider`.
    pub fn decide(&self, command: &RunCommand) -> Result<Vec<RunEvent>, &'static str> {
        match command {
            RunCommand::Admit { box_mode } => {
                if *box_mode == Mode::Queue {
                    Ok(vec![RunEvent::UnitAdmitted])
                } else {
                    Err("inv-box-must-be-queue")
                }
            }
            RunCommand::ComputeVerdict => {
                if self.phase == Phase::Admitted {
                    Ok(vec![RunEvent::UnitVerdictComputed])
                } else {
                    Err("inv-run-not-admitted")
                }
            }
            RunCommand::Emit => {
                if self.phase == Phase::VerdictReached {
                    Ok(vec![RunEvent::VerdictEmitted])
                } else {
                    Err("inv-no-verdict-yet")
                }
            }
        }
    }
}

/// The outcome of a workspace verification: pass/fail plus a machine-readable
/// evidence record (the check command, exit code, and output tails) that rides
/// on the VerdictEvent's cell-results for audit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateReport {
    pub passed: bool,
    pub evidence: Option<serde_json::Value>,
}

/// A protected oracle the cell-worker cannot write. It gates in two registers:
/// the in-memory per-cell `gate` (schema/structure, used by the demo walk), and
/// `verify`, which runs the real protected check against the artifacts a unit
/// wrote into its workspace — the gate that makes an `accepted` verdict mean
/// "it compiles and the tests pass," not merely "the model returned something."
pub trait Oracle {
    fn gate(&self, cell: &Cell) -> bool;

    /// Verify the produced artifacts sitting in `workspace`. The default is a
    /// pass-through (used by trivial/test oracles); a real oracle runs the
    /// project's protected check command with `workspace` as its working dir.
    fn verify(&self, _workspace: &Path) -> GateReport {
        GateReport { passed: true, evidence: None }
    }
}

/// Last `n` characters of a byte buffer, lossily decoded — for evidence tails.
fn tail(bytes: &[u8], n: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(n);
    chars[start..].iter().collect()
}

/// Walk the sealed cell-DAG: a cell is ready once its in-unit predecessors have
/// passed; gate each ready cell against the oracle. Returns the per-cell results
/// in dependency order. A blocked-forever cell (failed predecessor) is recorded
/// as not-passed without being gated.
pub fn walk_cells(cells: &[Cell], oracle: &dyn Oracle) -> Vec<CellResult> {
    use std::collections::BTreeMap;
    let mut passed: BTreeMap<String, bool> = BTreeMap::new();
    let mut results = Vec::new();
    // Cells are a DAG with intra-unit edges only; iterate to a fixpoint.
    let mut remaining: Vec<&Cell> = cells.iter().collect();
    while !remaining.is_empty() {
        let ready_idx = remaining.iter().position(|c| {
            c.depends_on.iter().all(|d| passed.contains_key(d))
        });
        let Some(idx) = ready_idx else {
            // A cycle or a dangling dependency — the rest were never gated.
            for c in &remaining {
                passed.insert(c.cell_id.clone(), false);
                results.push(CellResult::skipped(c.cell_id.clone()));
            }
            break;
        };
        let cell = remaining.remove(idx);
        let preds_ok = cell.depends_on.iter().all(|d| *passed.get(d).unwrap_or(&false));
        if !preds_ok {
            // A predecessor failed — this cell is skipped, not gated.
            passed.insert(cell.cell_id.clone(), false);
            results.push(CellResult::skipped(cell.cell_id.clone()));
        } else {
            let ok = oracle.gate(cell);
            passed.insert(cell.cell_id.clone(), ok);
            results.push(CellResult::gated(cell.cell_id.clone(), ok));
        }
    }
    results
}

/// Reduce cell-verdicts to a unit-verdict (`cell-verdict-rollup`): all-pass is
/// accepted; any failure escalates so the whole unit re-runs one binding up.
pub fn reduce_verdict(results: &[CellResult]) -> Verdict {
    if results.iter().all(|r| r.passed) {
        Verdict::Accepted
    } else {
        Verdict::Escalate
    }
}

/// The Transition Contract binding for a verdict. `not-admitted` binds to `halt`
/// — never `advance`: a higher tier cannot add a missing capability, so the
/// consumer re-routes to a covering box or surfaces to a human, it does not
/// advance the ladder against this executor.
pub fn consequence_of(verdict: Verdict) -> Consequence {
    match verdict {
        Verdict::Accepted => Consequence::Advance,
        Verdict::Rejected => Consequence::Halt,
        Verdict::Escalate => Consequence::Escalate,
        Verdict::NotAdmitted => Consequence::Halt,
    }
}

/// `verdict-stream-view` projector: the count of emitted verdicts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerdictStreamView {
    pub emitted: i64,
}

impl VerdictStreamView {
    pub fn apply(&mut self, _event: &RunEvent) {}
    pub fn on_emitted(&mut self) {
        self.emitted += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_interface::ModelBinding;
    use std::collections::BTreeMap;

    fn replay(events: &[RunEvent]) -> RunState {
        let mut s = RunState::default();
        for e in events { s.evolve(e); }
        s
    }

    #[test]
    fn admit_only_in_queue() {
        assert_eq!(replay(&[]).decide(&RunCommand::Admit { box_mode: Mode::Queue }), Ok(vec![RunEvent::UnitAdmitted]));
    }
    #[test]
    fn admit_refused_in_explorer() {
        assert_eq!(replay(&[]).decide(&RunCommand::Admit { box_mode: Mode::Explorer }), Err("inv-box-must-be-queue"));
    }
    #[test]
    fn verdict_computed_after_admit() {
        assert_eq!(replay(&[RunEvent::UnitAdmitted]).decide(&RunCommand::ComputeVerdict), Ok(vec![RunEvent::UnitVerdictComputed]));
    }
    #[test]
    fn verdict_not_computed_before_admit() {
        assert_eq!(replay(&[]).decide(&RunCommand::ComputeVerdict), Err("inv-run-not-admitted"));
    }
    #[test]
    fn emit_after_verdict() {
        assert_eq!(replay(&[RunEvent::UnitAdmitted, RunEvent::UnitVerdictComputed]).decide(&RunCommand::Emit), Ok(vec![RunEvent::VerdictEmitted]));
    }
    #[test]
    fn no_emit_before_verdict() {
        assert_eq!(replay(&[RunEvent::UnitAdmitted]).decide(&RunCommand::Emit), Err("inv-no-verdict-yet"));
    }

    struct AllPass;
    impl Oracle for AllPass {
        fn gate(&self, _c: &Cell) -> bool { true }
    }
    struct FailCell(&'static str);
    impl Oracle for FailCell {
        fn gate(&self, c: &Cell) -> bool { c.cell_id != self.0 }
    }

    fn cell(id: &str, deps: &[&str]) -> Cell {
        Cell {
            cell_id: id.into(),
            binding: ModelBinding { model: "coder".into(), quantization: "q4".into(), params: BTreeMap::new(), ..Default::default() },
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            prompt: format!("do {id}"),
            schema: serde_json::json!({ "type": "string" }),
            ..Default::default()
        }
    }

    #[test]
    fn all_cells_passing_accepts() {
        let cells = vec![cell("test", &[]), cell("impl", &["test"])];
        let r = walk_cells(&cells, &AllPass);
        assert_eq!(reduce_verdict(&r), Verdict::Accepted);
    }

    #[test]
    fn a_failed_cell_escalates_the_unit() {
        let cells = vec![cell("test", &[]), cell("impl", &["test"])];
        let r = walk_cells(&cells, &FailCell("impl"));
        assert_eq!(reduce_verdict(&r), Verdict::Escalate);
        assert_eq!(consequence_of(Verdict::Escalate), Consequence::Escalate);
    }

    #[test]
    fn a_failed_predecessor_blocks_its_successor() {
        // test-before-implement: if the test cell fails, impl is never gated green.
        let cells = vec![cell("test", &[]), cell("impl", &["test"])];
        let r = walk_cells(&cells, &FailCell("test"));
        let impl_passed = r.iter().find(|c| c.cell_id == "impl").unwrap().passed;
        assert!(!impl_passed);
    }
}

// ───────────────────────── oracle-run-decider ───────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleEvent {
    GateConfirmed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleCommand {
    /// `oracle_protected` = the cell-worker has no write capability over the oracle.
    RunGate { oracle_protected: bool },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OracleRunState {
    pub confirmed: bool,
}
impl OracleRunState {
    pub fn evolve(&mut self, e: &OracleEvent) {
        match e {
            OracleEvent::GateConfirmed => self.confirmed = true,
        }
    }
    /// `oracle-run-decider`: a gate may run only against a worker-unwritable
    /// oracle (ADR-076) — otherwise there is no independent verifier.
    pub fn decide(&self, c: &OracleCommand) -> Result<Vec<OracleEvent>, &'static str> {
        match c {
            OracleCommand::RunGate { oracle_protected } => {
                if *oracle_protected { Ok(vec![OracleEvent::GateConfirmed]) } else { Err("inv-oracle-writable") }
            }
        }
    }
}

/// A real protected oracle: gates a cell by running an external command (e.g.
/// `cargo test <filter>` or a check script) that lives outside, and is
/// unwritable by, the worker's workspace. Fails closed if its integrity is unmet.
pub struct CommandOracle {
    pub command: String,
    pub worker_writable: bool,
}
impl CommandOracle {
    /// The ADR-076 integrity check, as a decider outcome.
    pub fn integrity(&self) -> Result<Vec<OracleEvent>, &'static str> {
        OracleRunState::default().decide(&OracleCommand::RunGate { oracle_protected: !self.worker_writable })
    }
}
impl Oracle for CommandOracle {
    fn gate(&self, _cell: &Cell) -> bool {
        if self.integrity().is_err() {
            return false; // an oracle the worker can write is no oracle — fail closed
        }
        std::process::Command::new("sh")
            .arg("-c")
            .arg(&self.command)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Run the protected check command with `workspace` as its working directory,
    /// so it sees the artifacts this unit just wrote (e.g. `cargo test` against a
    /// seeded clone). Fails closed if the oracle's integrity (ADR-076) is unmet.
    fn verify(&self, workspace: &Path) -> GateReport {
        if self.integrity().is_err() {
            return GateReport {
                passed: false,
                evidence: Some(serde_json::json!({ "error": "oracle-writable", "command": self.command })),
            };
        }
        match std::process::Command::new("sh")
            .arg("-c")
            .arg(&self.command)
            .current_dir(workspace)
            .output()
        {
            Ok(o) => GateReport {
                passed: o.status.success(),
                evidence: Some(serde_json::json!({
                    "command": self.command,
                    "exit": o.status.code(),
                    "stdout_tail": tail(&o.stdout, 2000),
                    "stderr_tail": tail(&o.stderr, 2000),
                })),
            },
            Err(e) => GateReport {
                passed: false,
                evidence: Some(serde_json::json!({ "error": e.to_string(), "command": self.command })),
            },
        }
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use kiln_interface::ModelBinding;
    use std::collections::BTreeMap;

    fn cell() -> Cell {
        Cell {
            cell_id: "c".into(),
            binding: ModelBinding { model: "m".into(), quantization: "q".into(), params: BTreeMap::new(), ..Default::default() },
            prompt: "do c".into(),
            schema: serde_json::json!({ "type": "string" }),
            ..Default::default()
        }
    }

    #[test]
    fn gate_against_protected_oracle_is_confirmed() {
        assert_eq!(OracleRunState::default().decide(&OracleCommand::RunGate { oracle_protected: true }), Ok(vec![OracleEvent::GateConfirmed]));
    }
    #[test]
    fn gate_against_writable_oracle_is_refused() {
        assert_eq!(OracleRunState::default().decide(&OracleCommand::RunGate { oracle_protected: false }), Err("inv-oracle-writable"));
    }
    #[test]
    fn protected_command_oracle_runs_the_check() {
        let pass = CommandOracle { command: "true".into(), worker_writable: false };
        let fail = CommandOracle { command: "false".into(), worker_writable: false };
        assert!(pass.gate(&cell()));
        assert!(!fail.gate(&cell()));
    }
    #[test]
    fn worker_writable_oracle_fails_closed_even_if_command_passes() {
        let compromised = CommandOracle { command: "true".into(), worker_writable: true };
        assert!(!compromised.gate(&cell())); // integrity unmet -> never trusted
    }
}
