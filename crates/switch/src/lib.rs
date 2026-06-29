//! Box Control — the developer switch. The box has exactly one mode at a time;
//! the actuator enforces VRAM exclusion. Realises `box-decider`.

use serde::{Deserialize, Serialize};

/// The box configuration. `Off` is the cold state before any switch is thrown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Off,
    Queue,
    Explorer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoxEvent {
    ModeChanged { mode: Mode },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoxCommand {
    ThrowSwitch { mode: Mode },
}

/// The aggregate state the decider folds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxState {
    pub mode: Mode,
}

impl Default for BoxState {
    fn default() -> Self {
        BoxState { mode: Mode::Off }
    }
}

impl BoxState {
    pub fn evolve(&mut self, event: &BoxEvent) {
        match event {
            BoxEvent::ModeChanged { mode } => self.mode = *mode,
        }
    }

    /// `box-decider`: throwing the switch to the already-resident mode is a
    /// rejected no-op (inv-distinct-mode); otherwise it loads the new residency.
    pub fn decide(&self, command: &BoxCommand) -> Result<Vec<BoxEvent>, &'static str> {
        match command {
            BoxCommand::ThrowSwitch { mode } => {
                if *mode != self.mode {
                    Ok(vec![BoxEvent::ModeChanged { mode: *mode }])
                } else {
                    Err("inv-distinct-mode")
                }
            }
        }
    }
}

/// `box-status-view` projector: the box's current residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxStatusView {
    pub mode: Mode,
}

impl Default for BoxStatusView {
    fn default() -> Self {
        BoxStatusView { mode: Mode::Off }
    }
}

impl BoxStatusView {
    pub fn apply(&mut self, event: &BoxEvent) {
        match event {
            BoxEvent::ModeChanged { mode } => self.mode = *mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(events: &[BoxEvent]) -> BoxState {
        let mut s = BoxState::default();
        for e in events {
            s.evolve(e);
        }
        s
    }

    // --- box-decider scenarios (the oracle) ---
    #[test]
    fn first_throw_loads_queue_from_a_cold_box() {
        let s = replay(&[]);
        assert_eq!(s.decide(&BoxCommand::ThrowSwitch { mode: Mode::Queue }),
                   Ok(vec![BoxEvent::ModeChanged { mode: Mode::Queue }]));
    }

    #[test]
    fn flipping_queue_to_explorer_is_a_real_change() {
        let s = replay(&[BoxEvent::ModeChanged { mode: Mode::Queue }]);
        assert_eq!(s.decide(&BoxCommand::ThrowSwitch { mode: Mode::Explorer }),
                   Ok(vec![BoxEvent::ModeChanged { mode: Mode::Explorer }]));
    }

    #[test]
    fn rethrowing_the_resident_mode_is_rejected() {
        let s = replay(&[BoxEvent::ModeChanged { mode: Mode::Queue }]);
        assert_eq!(s.decide(&BoxCommand::ThrowSwitch { mode: Mode::Queue }), Err("inv-distinct-mode"));
    }

    // --- box-status-view projector scenario ---
    #[test]
    fn view_reflects_latest_residency() {
        let mut v = BoxStatusView::default();
        v.apply(&BoxEvent::ModeChanged { mode: Mode::Queue });
        assert_eq!(v.mode, Mode::Queue);
    }
}
