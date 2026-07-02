//! Model Serving — batched inference beneath QUEUE. Realises `resident-set-decider`,
//! `work-batch-decider`, the residency/throughput projectors, and the `Worker`
//! seam (the real model invocation, behind a trait so a GPU server plugs in).

use serde::{Deserialize, Serialize};
use spark_interface::{Cell, ModelBinding};

// ───────────────────────── resident-set-decider ─────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingEvent {
    BindingLoaded,
    BindingEvicted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingCommand {
    /// `within_budget` is the caller's VRAM-fit computation.
    Load { within_budget: bool },
    Evict,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentState {
    pub resident: bool,
}

impl ResidentState {
    pub fn evolve(&mut self, e: &BindingEvent) {
        match e {
            BindingEvent::BindingLoaded => self.resident = true,
            BindingEvent::BindingEvicted => self.resident = false,
        }
    }
    pub fn decide(&self, c: &BindingCommand) -> Result<Vec<BindingEvent>, &'static str> {
        match c {
            BindingCommand::Load { within_budget } => {
                if *within_budget { Ok(vec![BindingEvent::BindingLoaded]) } else { Err("inv-vram-budget") }
            }
            BindingCommand::Evict => {
                if self.resident { Ok(vec![BindingEvent::BindingEvicted]) } else { Err("inv-nothing-resident") }
            }
        }
    }
}

// ───────────────────────── work-batch-decider ───────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchPhase {
    New,
    Forming,
    Dispatched,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchEvent {
    BatchFormed,
    BatchDispatched,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchCommand {
    Form { homogeneous: bool, nonempty: bool },
    Dispatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchState {
    pub phase: BatchPhase,
}
impl Default for BatchState {
    fn default() -> Self {
        BatchState { phase: BatchPhase::New }
    }
}
impl BatchState {
    pub fn evolve(&mut self, e: &BatchEvent) {
        match e {
            BatchEvent::BatchFormed => self.phase = BatchPhase::Forming,
            BatchEvent::BatchDispatched => self.phase = BatchPhase::Dispatched,
        }
    }
    pub fn decide(&self, c: &BatchCommand) -> Result<Vec<BatchEvent>, &'static str> {
        match c {
            BatchCommand::Form { homogeneous, nonempty } => {
                if !*homogeneous { return Err("inv-batch-homogeneous"); }
                if !*nonempty { return Err("inv-batch-empty"); }
                Ok(vec![BatchEvent::BatchFormed])
            }
            BatchCommand::Dispatch => {
                if self.phase == BatchPhase::Forming { Ok(vec![BatchEvent::BatchDispatched]) } else { Err("inv-batch-not-formed") }
            }
        }
    }
}

/// Group ready cells by their required binding into homogeneous, non-empty
/// batches — the `batch-scheduler` pattern realising the dense-frontier principle.
pub fn schedule_batches(ready: &[Cell]) -> Vec<(ModelBinding, Vec<String>)> {
    use std::collections::BTreeMap;
    let mut by_binding: BTreeMap<String, (ModelBinding, Vec<String>)> = BTreeMap::new();
    for cell in ready {
        let entry = by_binding.entry(cell.binding.key()).or_insert_with(|| (cell.binding.clone(), Vec::new()));
        entry.1.push(cell.cell_id.clone());
    }
    by_binding.into_values().collect()
}

// ───────────────────────── projectors ───────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyView {
    pub resident: i64,
}
impl ResidencyView {
    pub fn apply(&mut self, e: &BindingEvent) {
        match e {
            BindingEvent::BindingLoaded => self.resident += 1,
            BindingEvent::BindingEvicted => self.resident -= 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchThroughputView {
    pub dispatched: i64,
}
impl BatchThroughputView {
    pub fn on_dispatched(&mut self) {
        self.dispatched += 1;
    }
}

// ───────────────────────── Worker seam ──────────────────────────────────

/// The real model invocation. A cell-worker is a function of its frozen prompt
/// alone (frozen-input). The GPU batching server is one implementation; tests
/// use the deterministic stub; production points `CommandWorker` at a served model.
pub trait Worker {
    fn invoke(&self, binding: &ModelBinding, prompt: &str) -> Result<String, String>;
}

/// Deterministic offline worker — echoes a structured response. Lets the whole
/// pipeline run and be tested with no model present.
pub struct StubWorker;
impl Worker for StubWorker {
    fn invoke(&self, binding: &ModelBinding, prompt: &str) -> Result<String, String> {
        Ok(format!("// produced by {} for: {}", binding.model, prompt.lines().next().unwrap_or("")))
    }
}

/// Shells out to an external served model (e.g. a llama.cpp / vLLM CLI) named by
/// `$SPARK_WORKER_CMD`, passing the prompt on stdin. A real, dependency-free seam.
pub struct CommandWorker {
    pub command: String,
}
impl Worker for CommandWorker {
    fn invoke(&self, _binding: &ModelBinding, prompt: &str) -> Result<String, String> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(&self.command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn worker: {e}"))?;
        child.stdin.take().unwrap().write_all(prompt.as_bytes()).map_err(|e| e.to_string())?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(format!("worker exited {}", out.status))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn replay_b(es: &[BindingEvent]) -> ResidentState {
        let mut s = ResidentState::default();
        for e in es { s.evolve(e); }
        s
    }
    fn replay_batch(es: &[BatchEvent]) -> BatchState {
        let mut s = BatchState::default();
        for e in es { s.evolve(e); }
        s
    }

    #[test]
    fn binding_that_fits_loads() {
        assert_eq!(replay_b(&[]).decide(&BindingCommand::Load { within_budget: true }), Ok(vec![BindingEvent::BindingLoaded]));
    }
    #[test]
    fn binding_over_budget_is_refused() {
        assert_eq!(replay_b(&[]).decide(&BindingCommand::Load { within_budget: false }), Err("inv-vram-budget"));
    }
    #[test]
    fn resident_binding_evicts() {
        assert_eq!(replay_b(&[BindingEvent::BindingLoaded]).decide(&BindingCommand::Evict), Ok(vec![BindingEvent::BindingEvicted]));
    }
    #[test]
    fn evicting_empty_is_refused() {
        assert_eq!(replay_b(&[]).decide(&BindingCommand::Evict), Err("inv-nothing-resident"));
    }
    #[test]
    fn homogeneous_nonempty_batch_forms() {
        assert_eq!(replay_batch(&[]).decide(&BatchCommand::Form { homogeneous: true, nonempty: true }), Ok(vec![BatchEvent::BatchFormed]));
    }
    #[test]
    fn mixed_batch_is_refused() {
        assert_eq!(replay_batch(&[]).decide(&BatchCommand::Form { homogeneous: false, nonempty: true }), Err("inv-batch-homogeneous"));
    }
    #[test]
    fn empty_batch_is_refused() {
        assert_eq!(replay_batch(&[]).decide(&BatchCommand::Form { homogeneous: true, nonempty: false }), Err("inv-batch-empty"));
    }
    #[test]
    fn formed_batch_dispatches() {
        assert_eq!(replay_batch(&[BatchEvent::BatchFormed]).decide(&BatchCommand::Dispatch), Ok(vec![BatchEvent::BatchDispatched]));
    }
    #[test]
    fn unformed_batch_refused() {
        assert_eq!(replay_batch(&[]).decide(&BatchCommand::Dispatch), Err("inv-batch-not-formed"));
    }
    #[test]
    fn scheduler_groups_by_binding() {
        let b = |q: &str| ModelBinding { model: "coder".into(), quantization: q.into(), params: BTreeMap::new(), ..Default::default() };
        let cells = vec![
            Cell { cell_id: "a".into(), binding: b("q4"), ..Default::default() },
            Cell { cell_id: "b".into(), binding: b("q4"), ..Default::default() },
            Cell { cell_id: "c".into(), binding: b("q8"), ..Default::default() },
        ];
        let batches = schedule_batches(&cells);
        assert_eq!(batches.len(), 2); // two distinct bindings -> two homogeneous batches
    }
    #[test]
    fn stub_worker_is_deterministic() {
        let b = ModelBinding { model: "m".into(), quantization: "q".into(), params: BTreeMap::new(), ..Default::default() };
        assert_eq!(StubWorker.invoke(&b, "do x").unwrap(), StubWorker.invoke(&b, "do x").unwrap());
    }
}

// ───────────────────── OpenAI-compatible HTTP Worker ────────────────────

/// A `Worker` that talks to an OpenAI-compatible chat-completions endpoint —
/// `llama-server`, vLLM, TGI, or any `/v1/chat/completions` server running on the
/// box. Dependency-free: plain HTTP/1.1 over a `TcpStream` (localhost needs no
/// TLS). One persistent server serves every cell, so there is no per-cell process
/// spawn — the batched-serving regime the box is built for.
pub struct OpenAiWorker {
    /// Server root, e.g. `http://localhost:8080` (no trailing `/v1`).
    pub base_url: String,
    /// Model name to request; empty → use the cell binding's model.
    pub model: String,
    /// Optional bearer token (vLLM `--api-key`, hosted gateways).
    pub api_key: Option<String>,
    /// Suppress a reasoning model's `<think>` phase (Qwen3 et al). With thinking
    /// ON, a reasoning model spends its budget in `message.reasoning` and returns
    /// `content: null` when it runs out — useless for artifact extraction. We send
    /// `chat_template_kwargs.enable_thinking = false` so the answer lands in
    /// `content`. Default true; set `SPARK_OPENAI_THINKING=on` to keep it.
    pub disable_thinking: bool,
    /// Token ceiling for the completion when the binding declares none. A reasoning
    /// model needs headroom; default 2048.
    pub max_tokens: i64,
}

impl OpenAiWorker {
    /// Build from `SPARK_OPENAI_BASE_URL` (+ optional `_MODEL`, `_API_KEY`,
    /// `_THINKING`, `_MAX_TOKENS`). Returns `None` when no base URL is set, so
    /// callers fall back to another worker.
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("SPARK_OPENAI_BASE_URL").ok()?;
        Some(OpenAiWorker {
            base_url,
            model: std::env::var("SPARK_OPENAI_MODEL").unwrap_or_default(),
            api_key: std::env::var("SPARK_OPENAI_API_KEY").ok(),
            disable_thinking: std::env::var("SPARK_OPENAI_THINKING").map(|v| v != "on").unwrap_or(true),
            max_tokens: std::env::var("SPARK_OPENAI_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048),
        })
    }

    /// Construct pointing at a materialized host endpoint, with env-tunable
    /// thinking/token defaults. Used by the CLI for the on-box vLLM residency.
    pub fn for_endpoint(base_url: String, model: String, api_key: Option<String>) -> Self {
        OpenAiWorker {
            base_url,
            model,
            api_key,
            disable_thinking: std::env::var("SPARK_OPENAI_THINKING").map(|v| v != "on").unwrap_or(true),
            max_tokens: std::env::var("SPARK_OPENAI_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048),
        }
    }
}

impl Worker for OpenAiWorker {
    fn invoke(&self, binding: &ModelBinding, prompt: &str) -> Result<String, String> {
        let model = if self.model.is_empty() { binding.model.clone() } else { self.model.clone() };
        let mut req = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "stream": false,
            "max_tokens": self.max_tokens,
        });
        // Map known binding params (strings on the Spark) onto request fields.
        if let Some(t) = binding.params.get("temperature").and_then(|v| v.parse::<f64>().ok()) {
            req["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = binding.params.get("top_p").and_then(|v| v.parse::<f64>().ok()) {
            req["top_p"] = serde_json::json!(p);
        }
        if let Some(k) = binding.params.get("top_k").and_then(|v| v.parse::<i64>().ok()) {
            req["top_k"] = serde_json::json!(k);
        }
        if let Some(n) = binding.params.get("max_tokens").and_then(|v| v.parse::<i64>().ok()) {
            req["max_tokens"] = serde_json::json!(n);
        }
        // Turn off the reasoning phase so the answer lands in `content`, not
        // `reasoning`. Harmless on non-reasoning servers (an unknown kwarg).
        if self.disable_thinking {
            req["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": false });
        }
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
        let body = http_post_json(&url, &req.to_string(), self.api_key.as_deref())?;
        let resp: serde_json::Value = serde_json::from_str(&body).map_err(|e| format!("parse response: {e}: {body}"))?;
        let msg = &resp["choices"][0]["message"];
        // Prefer `content`; if a reasoning model still emptied it into `reasoning`
        // (e.g. thinking left on), fall back so the run is not lost.
        let content = msg["content"].as_str().filter(|s| !s.is_empty());
        let reasoning = msg["reasoning"].as_str().filter(|s| !s.is_empty());
        match content.or(reasoning) {
            Some(s) => Ok(s.to_string()),
            None => {
                let finish = resp["choices"][0]["finish_reason"].as_str().unwrap_or("?");
                Err(format!("empty content (finish_reason={finish}); response: {body}"))
            }
        }
    }
}

/// Minimal blocking HTTP/1.1 `POST` with a JSON body over `TcpStream`. Sends
/// `Connection: close` and reads the response to EOF — no chunked-encoding parsing
/// needed. `http://` only (on-box localhost); for remote TLS, swap in a real client.
fn http_post_json(url: &str, json_body: &str, api_key: Option<&str>) -> Result<String, String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let rest = url.strip_prefix("http://").ok_or_else(|| format!("only http:// urls supported, got: {url}"))?;
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| format!("bad port in {host_port}"))?),
        None => (host_port, 80),
    };

    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    let auth = api_key.map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\n{auth}\
         Content-Length: {}\r\nConnection: close\r\n\r\n{json_body}",
        json_body.len()
    );
    stream.write_all(request.as_bytes()).map_err(|e| format!("write: {e}"))?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| format!("read: {e}"))?;
    let text = String::from_utf8_lossy(&raw);
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    let status_ok = text.lines().next().map(|l| l.contains(" 200")).unwrap_or(false);
    if !status_ok {
        return Err(format!("non-200 from {host}:{port}: {}", text.lines().next().unwrap_or("")));
    }
    Ok(body.to_string())
}

#[cfg(test)]
mod http_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A real round-trip against a one-shot mock OpenAI server on an ephemeral
    /// port — proves the HTTP path end to end with no external service.
    #[test]
    fn openai_worker_round_trips_against_a_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf); // consume request line/headers/body
            let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n\
                        {\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"fn add(a:i32,b:i32)->i32{a+b}\"}}]}";
            sock.write_all(resp.as_bytes()).unwrap();
        });

        let worker = OpenAiWorker::for_endpoint(format!("http://127.0.0.1:{port}"), "test-model".into(), None);
        let binding = ModelBinding { model: "coder".into(), quantization: "q4".into(), params: BTreeMap::new(), ..Default::default() };
        let out = worker.invoke(&binding, "write add()").unwrap();
        assert_eq!(out, "fn add(a:i32,b:i32)->i32{a+b}");
        handle.join().unwrap();
    }

    #[test]
    fn rejects_non_http_scheme() {
        assert!(http_post_json("https://example.com/v1", "{}", None).is_err());
    }
}
