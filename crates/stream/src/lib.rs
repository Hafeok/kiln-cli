//! Verdict Stream — the durable outbound seam. Realises `verdict-log-decider`,
//! the reconciliation projector, and a real append-only, idempotent log on disk.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use spark_interface::VerdictEvent;

// ───────────────────────── verdict-log-decider ──────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogEvent {
    VerdictAppended,
    DeliverableReconciled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogCommand {
    /// `already_logged` = a verdict for this bundle_hash is already present.
    Append { already_logged: bool },
    Reconcile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogState {
    pub pending: i64,
}
impl LogState {
    pub fn evolve(&mut self, e: &LogEvent) {
        match e {
            LogEvent::VerdictAppended => self.pending += 1,
            LogEvent::DeliverableReconciled => self.pending = 0,
        }
    }
    pub fn decide(&self, c: &LogCommand) -> Result<Vec<LogEvent>, &'static str> {
        match c {
            LogCommand::Append { already_logged } => {
                if *already_logged { Err("inv-idempotent-append") } else { Ok(vec![LogEvent::VerdictAppended]) }
            }
            LogCommand::Reconcile => {
                if self.pending > 0 { Ok(vec![LogEvent::DeliverableReconciled]) } else { Err("inv-nothing-to-reconcile") }
            }
        }
    }
}

// ───────────────────────── reconciliation projector ─────────────────────

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationView {
    pub appended: i64,
    pub reconciled: i64,
}
impl ReconciliationView {
    pub fn apply(&mut self, e: &LogEvent) {
        match e {
            LogEvent::VerdictAppended => self.appended += 1,
            LogEvent::DeliverableReconciled => self.reconciled += 1,
        }
    }
    pub fn backlog(&self) -> i64 {
        self.appended - self.reconciled
    }
}

// ───────────────────────── durable append-only log ──────────────────────

/// A real append-only, idempotent log of VerdictEvents on disk (JSON lines).
/// Idempotent by `bundle_hash`: a re-emitted verdict is not appended twice, so
/// at-least-once emission never doubles a record. Survives restart by re-reading.
pub struct DurableLog {
    path: PathBuf,
    seen: BTreeSet<String>,
}

impl DurableLog {
    /// Open (or create) the log at `path`, recovering the set of seen hashes so
    /// idempotency holds across restarts.
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let mut seen = BTreeSet::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<VerdictEvent>(line) {
                    seen.insert(v.bundle_hash);
                }
            }
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(DurableLog { path, seen })
    }

    /// Append a verdict; returns true if newly appended, false if a verdict for
    /// this bundle_hash was already present (the idempotency guard's data side).
    pub fn append(&mut self, verdict: &VerdictEvent) -> std::io::Result<bool> {
        if self.seen.contains(&verdict.bundle_hash) {
            return Ok(false);
        }
        let line = serde_json::to_string(verdict).expect("serialize verdict");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{line}")?;
        self.seen.insert(verdict.bundle_hash.clone());
        Ok(true)
    }

    pub fn already_logged(&self, bundle_hash: &str) -> bool {
        self.seen.contains(bundle_hash)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Read all verdicts back from disk (a consumer reconciling from cold).
    pub fn read_all(path: &Path) -> std::io::Result<Vec<VerdictEvent>> {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        Ok(text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_interface::{Consequence, Verdict};

    fn replay(es: &[LogEvent]) -> LogState {
        let mut s = LogState::default();
        for e in es { s.evolve(e); }
        s
    }
    fn verdict(hash: &str) -> VerdictEvent {
        VerdictEvent {
            event_id: format!("ve-{hash}"),
            emitted_at: "t0".into(),
            unit_ref: "u".into(),
            parent_deliverable: "d".into(),
            bundle_hash: hash.into(),
            verdict: Verdict::Accepted,
            missing_capabilities: None,
            tier_ran: Some("light".into()),
            cell_results: vec![],
            delivery_result: None,
            next_consequence: Consequence::Advance,
        }
    }

    #[test]
    fn fresh_verdict_appends() {
        assert_eq!(replay(&[]).decide(&LogCommand::Append { already_logged: false }), Ok(vec![LogEvent::VerdictAppended]));
    }
    #[test]
    fn duplicate_verdict_refused() {
        assert_eq!(replay(&[]).decide(&LogCommand::Append { already_logged: true }), Err("inv-idempotent-append"));
    }
    #[test]
    fn pending_reconciles() {
        assert_eq!(replay(&[LogEvent::VerdictAppended]).decide(&LogCommand::Reconcile), Ok(vec![LogEvent::DeliverableReconciled]));
    }
    #[test]
    fn empty_backlog_refused() {
        assert_eq!(replay(&[]).decide(&LogCommand::Reconcile), Err("inv-nothing-to-reconcile"));
    }

    #[test]
    fn durable_log_is_idempotent_and_survives_reopen() {
        let dir = std::env::temp_dir().join("spark-test-stream");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("verdicts.jsonl");
        {
            let mut log = DurableLog::open(&path).unwrap();
            assert!(log.append(&verdict("h1")).unwrap()); // new
            assert!(!log.append(&verdict("h1")).unwrap()); // idempotent dup
            assert!(log.append(&verdict("h2")).unwrap());
            assert_eq!(log.len(), 2);
        }
        // Reopen from disk: seen-set recovered, so a re-emit is still a dup.
        let mut log = DurableLog::open(&path).unwrap();
        assert!(log.already_logged("h1"));
        assert!(!log.append(&verdict("h1")).unwrap());
        assert_eq!(DurableLog::read_all(&path).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
