//! Exploration — the local explorer. Active only while the box is in EXPLORER;
//! its product is a discovery record, never accepted code. Realises
//! `exploration-session-decider`.

use spark_switch::Mode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Active,
    Complete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionEvent {
    ExplorationStarted,
    DiscoveryRecordProduced,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionCommand {
    Start { box_mode: Mode },
    ProduceDiscoveryRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionState {
    pub phase: Phase,
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState { phase: Phase::Idle }
    }
}

impl SessionState {
    pub fn evolve(&mut self, event: &SessionEvent) {
        match event {
            SessionEvent::ExplorationStarted => self.phase = Phase::Active,
            SessionEvent::DiscoveryRecordProduced => self.phase = Phase::Complete,
        }
    }

    /// `exploration-session-decider`.
    pub fn decide(&self, command: &SessionCommand) -> Result<Vec<SessionEvent>, &'static str> {
        match command {
            SessionCommand::Start { box_mode } => {
                if *box_mode == Mode::Explorer {
                    Ok(vec![SessionEvent::ExplorationStarted])
                } else {
                    Err("inv-box-must-be-explorer")
                }
            }
            SessionCommand::ProduceDiscoveryRecord => {
                if self.phase == Phase::Active {
                    Ok(vec![SessionEvent::DiscoveryRecordProduced])
                } else {
                    Err("inv-session-not-active")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(events: &[SessionEvent]) -> SessionState {
        let mut s = SessionState::default();
        for e in events { s.evolve(e); }
        s
    }

    #[test]
    fn session_starts_only_in_explorer() {
        assert_eq!(replay(&[]).decide(&SessionCommand::Start { box_mode: Mode::Explorer }),
                   Ok(vec![SessionEvent::ExplorationStarted]));
    }
    #[test]
    fn session_cannot_start_in_queue() {
        assert_eq!(replay(&[]).decide(&SessionCommand::Start { box_mode: Mode::Queue }), Err("inv-box-must-be-explorer"));
    }
    #[test]
    fn active_session_produces_a_record() {
        assert_eq!(replay(&[SessionEvent::ExplorationStarted]).decide(&SessionCommand::ProduceDiscoveryRecord),
                   Ok(vec![SessionEvent::DiscoveryRecordProduced]));
    }
    #[test]
    fn idle_session_cannot_produce() {
        assert_eq!(replay(&[]).decide(&SessionCommand::ProduceDiscoveryRecord), Err("inv-session-not-active"));
    }
}
