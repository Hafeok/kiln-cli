//! Live integration test: exercise a real Spark box end-to-end and prove it
//! produces **working C# code**.
//!
//! This is not a unit test with a stub — it drives the full production seam:
//!
//!   admit(WorkUnit) → per-unit sandbox seeded from a real .NET project
//!                   → the box's served model writes `Calc/StringOps.cs`
//!                   → the PROTECTED oracle `dotnet test` gates the artifact
//!                   → verdict.
//!
//! The xUnit test lives in the seed (`examples/csharp-seed/Calc.Tests`) and the
//! cell-worker has no write capability over it (ADR-076). So an `accepted`
//! verdict means the box wrote C# that *compiles and passes a contract it could
//! not author* — the strongest sense of "working code".
//!
//! Opt-in (network + `dotnet` required), so it is `#[ignore]`d out of the normal
//! `cargo test` run. Invoke explicitly:
//!
//!   cargo test -p spark-executor --test csharp_live -- --ignored --nocapture
//!
//! Configurable via env (defaults target the box on the bench):
//!   SPARK_OPENAI_BASE_URL   default http://192.168.88.63:8000
//!   SPARK_OPENAI_MODEL      default qwen3.6-35b
//!   SPARK_OPENAI_API_KEY    optional bearer token

use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::time::Duration;

use spark_execution::CommandOracle;
use spark_executor::{DrainOutcome, Engine};
use spark_interface::{Verdict, WorkUnit};
use spark_sandbox::{LocalBroker, LocalSandbox};
use spark_serving::OpenAiWorker;
use spark_stream::DurableLog;
use spark_switch::Mode;

const DEFAULT_BASE_URL: &str = "http://192.168.88.63:8000";
const DEFAULT_MODEL: &str = "qwen3.6-35b";

/// Repo root, derived from this crate's manifest dir (`crates/executor`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Parse `http://host:port` into a `(host, port)` for a reachability probe.
fn host_port(base_url: &str) -> (String, u16) {
    let rest = base_url.strip_prefix("http://").unwrap_or(base_url);
    let authority = rest.split('/').next().unwrap_or(rest);
    match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
        None => (authority.to_string(), 80),
    }
}

/// Fail loudly (not silently skip) when a prerequisite is missing: the test is
/// opt-in, so if you asked for it, an unmet precondition is a real problem.
fn require(cond: bool, msg: &str) {
    assert!(cond, "precondition unmet: {msg}");
}

#[test]
#[ignore = "live: needs the Spark box reachable and `dotnet` on PATH"]
fn box_writes_working_csharp() {
    let base_url = std::env::var("SPARK_OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
    let model = std::env::var("SPARK_OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let api_key = std::env::var("SPARK_OPENAI_API_KEY").ok();

    // --- Preconditions -----------------------------------------------------
    // `dotnet` present (the oracle shells out to it).
    let dotnet_ok = std::process::Command::new("dotnet")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    require(dotnet_ok, "`dotnet` not on PATH — install the .NET SDK");

    // The box answers on its TCP endpoint.
    let (host, port) = host_port(&base_url);
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next());
    let reachable = addr
        .and_then(|a| TcpStream::connect_timeout(&a, Duration::from_secs(3)).ok())
        .is_some();
    require(reachable, &format!("Spark box unreachable at {base_url} — is vLLM serving?"));

    let seed = repo_root().join("examples").join("csharp-seed");
    require(seed.join("Calc.Tests/SlugifyTests.cs").exists(), "seed project missing");

    // --- Arrange -----------------------------------------------------------
    // The WorkUnit is the canonical contract JSON, admitted through the real
    // homogeneity guard — same artifact `spark admit` consumes.
    let unit_json = std::fs::read_to_string(repo_root().join("examples").join("workunit-csharp.json"))
        .expect("read workunit-csharp.json");
    let unit: WorkUnit = WorkUnit::from_canonical_json(&unit_json).expect("parse canonical WorkUnit");
    let unit_ref = unit.unit_ref.clone();

    let mut engine = Engine::default();
    engine.throw_switch(Mode::Queue).expect("switch to QUEUE");
    engine.admit(unit).expect("admit binding-homogeneous unit");

    // Isolate all on-disk state under this test binary's temp dir.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("csharp_live");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp root");
    let sandbox = LocalSandbox { root: tmp.join("sandboxes") };
    let broker = LocalBroker;
    let mut log = DurableLog::open(tmp.join("verdicts.jsonl")).expect("open durable log");
    let deliveries = tmp.join("deliveries");

    // The real HTTP worker → the box; the PROTECTED oracle → `dotnet test`.
    let worker = OpenAiWorker::for_endpoint(base_url.clone(), model.clone(), api_key);
    let oracle = CommandOracle {
        command: "dotnet test Calc.Tests/Calc.Tests.csproj --nologo".into(),
        worker_writable: false,
    };

    // --- Act ---------------------------------------------------------------
    let outcome = engine
        .drain_one_isolated(
            &sandbox,
            &broker,
            &worker,
            &oracle,
            &mut log,
            Some(seed.as_path()),
            Some(deliveries.as_path()),
            "t-csharp-live",
        )
        .expect("isolated drain runs");

    // --- Assert ------------------------------------------------------------
    // The decisive claim: the box's C# compiled AND passed the protected xUnit
    // suite. Anything else (rejected → escalate/halt) fails the test.
    assert!(
        matches!(&outcome, DrainOutcome::Accepted { unit_ref: r } if *r == unit_ref),
        "expected Accepted, got {outcome:?} — the box's C# did not compile-and-pass `dotnet test`"
    );

    // The verdict is durably logged, and its recorded verdict is Accepted.
    assert_eq!(log.len(), 1, "exactly one verdict appended");
    let event = engine.stream.last().expect("a verdict event on the stream");
    assert_eq!(event.verdict, Verdict::Accepted, "logged verdict is Accepted");
    assert_eq!(event.unit_ref, unit_ref);

    // A reviewable patch of the generated C# was delivered.
    let safe: String = unit_ref
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let patch = deliveries.join(format!("{safe}.patch"));
    assert!(patch.exists(), "expected a delivered patch at {}", patch.display());
    let patch_text = std::fs::read_to_string(&patch).unwrap_or_default();
    assert!(
        patch_text.contains("Calc/StringOps.cs") && patch_text.contains("Slugify"),
        "delivered patch should contain the generated Slugify implementation"
    );

    eprintln!("✓ {unit_ref}: the Spark box wrote C# that compiled and passed the protected `dotnet test` gate");
}
