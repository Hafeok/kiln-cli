//! Isolation — per-unit ephemeral sandbox + brokered credentials (Environment +
//! Credentials). Realises `sandbox-decider`, `credential-lease-decider`, the
//! isolation projector, and the `SandboxRuntime` / `CredentialBroker` seams.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ───────────────────────── sandbox-decider ──────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxPhase {
    None,
    Provisioned,
    Destroyed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxEvent {
    SandboxProvisioned,
    SandboxDestroyed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxCommand {
    /// `network_declared` = every effect targets a declared destination.
    Provision { network_declared: bool },
    Teardown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxState {
    pub phase: SandboxPhase,
}
impl Default for SandboxState {
    fn default() -> Self {
        SandboxState { phase: SandboxPhase::None }
    }
}
impl SandboxState {
    pub fn evolve(&mut self, e: &SandboxEvent) {
        match e {
            SandboxEvent::SandboxProvisioned => self.phase = SandboxPhase::Provisioned,
            SandboxEvent::SandboxDestroyed => self.phase = SandboxPhase::Destroyed,
        }
    }
    pub fn decide(&self, c: &SandboxCommand) -> Result<Vec<SandboxEvent>, &'static str> {
        match c {
            SandboxCommand::Provision { network_declared } => {
                if *network_declared { Ok(vec![SandboxEvent::SandboxProvisioned]) } else { Err("inv-undeclared-network") }
            }
            SandboxCommand::Teardown => {
                if self.phase == SandboxPhase::Provisioned { Ok(vec![SandboxEvent::SandboxDestroyed]) } else { Err("inv-no-sandbox") }
            }
        }
    }
}

// ───────────────────────── credential-lease-decider ─────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeasePhase {
    None,
    Leased,
    Revoked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseEvent {
    CredentialLeased,
    CredentialRevoked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseCommand {
    Exchange { sandbox_active: bool },
    Revoke,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseState {
    pub phase: LeasePhase,
}
impl Default for LeaseState {
    fn default() -> Self {
        LeaseState { phase: LeasePhase::None }
    }
}
impl LeaseState {
    pub fn evolve(&mut self, e: &LeaseEvent) {
        match e {
            LeaseEvent::CredentialLeased => self.phase = LeasePhase::Leased,
            LeaseEvent::CredentialRevoked => self.phase = LeasePhase::Revoked,
        }
    }
    pub fn decide(&self, c: &LeaseCommand) -> Result<Vec<LeaseEvent>, &'static str> {
        match c {
            LeaseCommand::Exchange { sandbox_active } => {
                if *sandbox_active { Ok(vec![LeaseEvent::CredentialLeased]) } else { Err("inv-lease-needs-sandbox") }
            }
            LeaseCommand::Revoke => {
                if self.phase == LeasePhase::Leased { Ok(vec![LeaseEvent::CredentialRevoked]) } else { Err("inv-nothing-to-revoke") }
            }
        }
    }
}

// ───────────────────────── isolation projector ──────────────────────────

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsolationView {
    pub live: i64,
}
impl IsolationView {
    pub fn apply(&mut self, e: &SandboxEvent) {
        match e {
            SandboxEvent::SandboxProvisioned => self.live += 1,
            SandboxEvent::SandboxDestroyed => self.live -= 1,
        }
    }
}

// ───────────────────────── runtime seams ────────────────────────────────

/// The execution boundary. `LocalSandbox` gives a real per-unit directory; a
/// container/microVM runtime is a drop-in implementation for hard isolation.
pub trait SandboxRuntime {
    fn provision(&self, unit_ref: &str) -> std::io::Result<PathBuf>;
    fn teardown(&self, workspace: &Path) -> std::io::Result<()>;
}

/// A real, filesystem-backed boundary: a private writable workspace per unit
/// under `.kiln/sandboxes/`, destroyed (removed) at teardown.
pub struct LocalSandbox {
    pub root: PathBuf,
}
impl Default for LocalSandbox {
    fn default() -> Self {
        LocalSandbox { root: PathBuf::from(".kiln/sandboxes") }
    }
}
impl SandboxRuntime for LocalSandbox {
    fn provision(&self, unit_ref: &str) -> std::io::Result<PathBuf> {
        let safe: String = unit_ref.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
        let ws = self.root.join(safe);
        std::fs::create_dir_all(&ws)?;
        Ok(ws)
    }
    fn teardown(&self, workspace: &Path) -> std::io::Result<()> {
        if workspace.exists() {
            std::fs::remove_dir_all(workspace)?;
        }
        Ok(())
    }
}

/// Exchanges a grant-reference for a short-lived credential and revokes it. A
/// real broker (Vault, cloud STS) is a drop-in; `LocalBroker` mints opaque tokens.
pub trait CredentialBroker {
    fn exchange(&self, grant_ref: &str) -> String;
    fn revoke(&self, lease: &str);
}

pub struct LocalBroker;
impl CredentialBroker for LocalBroker {
    fn exchange(&self, grant_ref: &str) -> String {
        format!("lease:{grant_ref}")
    }
    fn revoke(&self, _lease: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay_s(es: &[SandboxEvent]) -> SandboxState {
        let mut s = SandboxState::default();
        for e in es { s.evolve(e); }
        s
    }
    fn replay_l(es: &[LeaseEvent]) -> LeaseState {
        let mut s = LeaseState::default();
        for e in es { s.evolve(e); }
        s
    }

    #[test]
    fn declared_network_provisions() {
        assert_eq!(replay_s(&[]).decide(&SandboxCommand::Provision { network_declared: true }), Ok(vec![SandboxEvent::SandboxProvisioned]));
    }
    #[test]
    fn undeclared_network_refused() {
        assert_eq!(replay_s(&[]).decide(&SandboxCommand::Provision { network_declared: false }), Err("inv-undeclared-network"));
    }
    #[test]
    fn provisioned_tears_down() {
        assert_eq!(replay_s(&[SandboxEvent::SandboxProvisioned]).decide(&SandboxCommand::Teardown), Ok(vec![SandboxEvent::SandboxDestroyed]));
    }
    #[test]
    fn teardown_without_sandbox_refused() {
        assert_eq!(replay_s(&[]).decide(&SandboxCommand::Teardown), Err("inv-no-sandbox"));
    }
    #[test]
    fn lease_for_live_sandbox() {
        assert_eq!(replay_l(&[]).decide(&LeaseCommand::Exchange { sandbox_active: true }), Ok(vec![LeaseEvent::CredentialLeased]));
    }
    #[test]
    fn lease_without_sandbox_refused() {
        assert_eq!(replay_l(&[]).decide(&LeaseCommand::Exchange { sandbox_active: false }), Err("inv-lease-needs-sandbox"));
    }
    #[test]
    fn active_lease_revokes() {
        assert_eq!(replay_l(&[LeaseEvent::CredentialLeased]).decide(&LeaseCommand::Revoke), Ok(vec![LeaseEvent::CredentialRevoked]));
    }
    #[test]
    fn revoke_without_lease_refused() {
        assert_eq!(replay_l(&[]).decide(&LeaseCommand::Revoke), Err("inv-nothing-to-revoke"));
    }
    #[test]
    fn local_sandbox_provisions_and_tears_down_a_real_dir() {
        let sb = LocalSandbox { root: std::env::temp_dir().join("kiln-test-sb") };
        let ws = sb.provision("wu-1/x").unwrap();
        assert!(ws.exists());
        sb.teardown(&ws).unwrap();
        assert!(!ws.exists());
    }
}
