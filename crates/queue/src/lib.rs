//! Work-Unit Queue — the flat, edgeless priority set and the admission guard.
//! Realises `work-unit-decider`, `priority-set-view`, `heterogeneity-rate-view`.

use serde::{Deserialize, Serialize};

use spark_interface::WorkUnit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    New,
    Queued,
    Rejected,
    Escalated,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitEvent {
    Enqueued,
    Rejected,
    Reprioritized,
    Escalated,
    Halted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitCommand {
    /// `homogeneous` is the admission re-check result (WorkUnit::is_binding_homogeneous).
    Admit { homogeneous: bool },
    Reprioritize,
    Escalate { max_ladder: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitState {
    pub state: Lifecycle,
    pub ladder: u32,
}

impl Default for UnitState {
    fn default() -> Self {
        UnitState { state: Lifecycle::New, ladder: 0 }
    }
}

impl UnitState {
    pub fn evolve(&mut self, event: &UnitEvent) {
        match event {
            UnitEvent::Enqueued => self.state = Lifecycle::Queued,
            UnitEvent::Reprioritized => self.state = Lifecycle::Queued,
            UnitEvent::Escalated => {
                self.state = Lifecycle::Queued;
                self.ladder += 1;
            }
            UnitEvent::Rejected => self.state = Lifecycle::Rejected,
            UnitEvent::Halted => self.state = Lifecycle::Failed,
        }
    }

    /// `work-unit-decider`.
    pub fn decide(&self, command: &UnitCommand) -> Result<Vec<UnitEvent>, &'static str> {
        match command {
            UnitCommand::Admit { homogeneous } => {
                if *homogeneous {
                    Ok(vec![UnitEvent::Enqueued])
                } else {
                    Err("inv-binding-homogeneity")
                }
            }
            UnitCommand::Reprioritize => {
                if self.state == Lifecycle::Queued {
                    Ok(vec![UnitEvent::Reprioritized])
                } else {
                    Err("inv-only-queued-reorderable")
                }
            }
            UnitCommand::Escalate { max_ladder } => {
                if self.ladder < *max_ladder {
                    Ok(vec![UnitEvent::Escalated])
                } else {
                    Err("inv-ladder-exhausted")
                }
            }
        }
    }
}

/// Decide admission for a concrete frozen unit. Runs the full contracts-layer
/// structural validation (`WorkUnit::validate`: binding-homogeneity, no cross-unit
/// `requires` edge, every `context_refs` id resolving, every cell executable) and
/// rejects any non-conforming unit with its specific invariant id before enqueue.
/// This is the "validate against the normative schema + structural check, reject
/// non-conforming units" consumer obligation, re-run defensively at admission.
pub fn decide_admission(unit: &WorkUnit) -> Result<Vec<UnitEvent>, &'static str> {
    unit.validate()?;
    UnitState::default().decide(&UnitCommand::Admit { homogeneous: true })
}

/// `priority-set-view` projector: the size of the flat queued set.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrioritySetView {
    pub queued: i64,
}

impl PrioritySetView {
    /// Fold a queue-owned event: enqueue and escalation (re-enqueue) add to the
    /// set; reprioritize and halt leave its size unchanged.
    pub fn on_queue_event(&mut self, event: &UnitEvent) {
        match event {
            UnitEvent::Enqueued | UnitEvent::Escalated => self.queued += 1,
            _ => {}
        }
    }
    /// A unit admitted into the executor leaves the set (the cross-context
    /// `unit-admitted` fold).
    pub fn on_admitted(&mut self) {
        self.queued -= 1;
    }
}

/// `heterogeneity-rate-view` projector: the maturation metric.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneityRateView {
    pub admitted: i64,
    pub rejected: i64,
}

impl HeterogeneityRateView {
    pub fn apply(&mut self, event: &UnitEvent) {
        match event {
            UnitEvent::Enqueued => self.admitted += 1,
            UnitEvent::Rejected => self.rejected += 1,
            _ => {}
        }
    }
    /// Rejected / total admissions; 0.0 when nothing has been admitted.
    pub fn rate(&self) -> f64 {
        let total = self.admitted + self.rejected;
        if total == 0 { 0.0 } else { self.rejected as f64 / total as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(events: &[UnitEvent]) -> UnitState {
        let mut s = UnitState::default();
        for e in events { s.evolve(e); }
        s
    }

    #[test]
    fn homogeneous_unit_is_enqueued() {
        assert_eq!(replay(&[]).decide(&UnitCommand::Admit { homogeneous: true }), Ok(vec![UnitEvent::Enqueued]));
    }
    #[test]
    fn heterogeneous_unit_is_rejected() {
        assert_eq!(replay(&[]).decide(&UnitCommand::Admit { homogeneous: false }), Err("inv-binding-homogeneity"));
    }
    #[test]
    fn queued_unit_can_be_reprioritized() {
        assert_eq!(replay(&[UnitEvent::Enqueued]).decide(&UnitCommand::Reprioritize), Ok(vec![UnitEvent::Reprioritized]));
    }
    #[test]
    fn unqueued_unit_cannot_be_reprioritized() {
        assert_eq!(replay(&[]).decide(&UnitCommand::Reprioritize), Err("inv-only-queued-reorderable"));
    }
    #[test]
    fn unit_with_headroom_escalates() {
        assert_eq!(replay(&[UnitEvent::Enqueued]).decide(&UnitCommand::Escalate { max_ladder: 3 }), Ok(vec![UnitEvent::Escalated]));
    }
    #[test]
    fn exhausted_ladder_halts() {
        assert_eq!(replay(&[UnitEvent::Enqueued]).decide(&UnitCommand::Escalate { max_ladder: 0 }), Err("inv-ladder-exhausted"));
    }

    #[test]
    fn priority_set_tracks_queue_size() {
        let mut v = PrioritySetView::default();
        for e in [UnitEvent::Enqueued, UnitEvent::Reprioritized] { v.on_queue_event(&e); }
        assert_eq!(v.queued, 1);
    }
    #[test]
    fn heterogeneity_rate_counts_both() {
        let mut v = HeterogeneityRateView::default();
        for e in [UnitEvent::Enqueued, UnitEvent::Enqueued, UnitEvent::Rejected] { v.apply(&e); }
        assert_eq!((v.admitted, v.rejected), (2, 1));
    }
}
