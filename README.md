# kiln-cli

A Rust implementation of **Kiln** (the execution framework, formerly the Spark
Execution Framework) — the execution pillar for running autonomous AI development
work, with the NVIDIA DGX Spark as its reference substrate.

`kiln-cli` is the developer switch and executor loop. It **publishes a
CapabilityManifest** (what can run here), admits frozen **WorkUnits**, runs their
sealed cell-DAGs through a verification gate, and emits **VerdictEvents** —
*computed*-done, never *claimed*-done. The whole thing is
spec-driven: the domain model (the **What**), the architecture (the **How**), and
the behavioural conformance suite all live under `.product/` and are kept in lockstep
with the code.

---

## The box has one switch

The Spark is bandwidth-bound and its VRAM holds **one residency at a time**. The
developer throws a switch between two mutually exclusive modes:

| Mode | What runs | Shape |
|---|---|---|
| **QUEUE** | many small models, batched inference | high-throughput, many units in flight |
| **EXPLORER** | one large model, serial | deep, single discovery session |
| *OFF* | nothing resident | the default |

The flip is a deliberate **human act** — no machine-rate process triggers it,
because swapping a residency is expensive. (Opus-class *OFF-BOX* work is not a box
mode; it lives elsewhere.)

---

## Core concepts

- **WorkUnit** — travels in by-value, identified by `bundle_hash`. Its cells form a
  sealed DAG. **Binding-homogeneity invariant**: every cell of a unit requires the
  *same* SPMC model binding (model identity + served quantization + params).
  Quantization is load-bearing on the Spark; a heterogeneous unit is a decomposition
  defect and is never dispatched.
- **CapabilityManifest** — the executor **publishes** its self-description out of
  band (`kiln manifest`): the `bindings` it can serve, the `delivery` modes/schemes/
  integration-methods/forges, the `shape-languages` and `gate-kinds` it runs. A
  producer matches a unit's derived *requirements* against it before dispatch —
  empty distance ⇒ executable. It is the one seam artifact the executor authors.
- **Admission** — a distinct gate *before* verification. A unit whose derived
  requirements the box cannot cover **never runs** — it is answered `not-admitted`
  with the concrete `missing-capabilities` distance, binding to `halt` (a higher
  tier never adds a missing capability).
- **Verification** — each cell is gated by a **protected oracle** the worker *cannot
  write* (ADR-076). A worker that can write its own oracle has no verifier. Verdicts
  are `accepted` / `rejected` / `escalate` / `not-admitted`; consequences are
  `advance` / `halt` / `retry` / `escalate`. Escalation is **unit-atomic** — the
  whole unit moves one binding up the ladder, never a single cell.
- **Artifact delivery** — a unit declares where produced work lands: `inline`
  (artifact bodies by value in the verdict) or `repository` (a declared git repo —
  `file:///` local, remote for production — landed via `push-branch` or
  `pull-request`; the verdict carries `delivery-result`: branch, commit, `pr-url`).
  No credential material ever rides in the WorkUnit — repository/forge access is
  exchanged executor-side from the `credential-grant` reference.
- **VerdictEvent** — travels out fire-and-forget to a durable, append-only log.

---

## Architecture (11 crates)

```
interface     WorkUnit / Cell / ModelBinding / VerdictEvent / CapabilityManifest  (by-value contract, bundle_hash)
switch        Box Control     — the developer switch (QUEUE ⇄ EXPLORER), distinct-mode guard
queue         Work-Unit Queue — admission (homogeneity guard), priority, escalation ladder
execution     Execution       — sealed cell-DAG walk, verdict reduction, + oracle-run gate
exploration   Exploration     — single serial discovery session (EXPLORER)
serving       Model Serving   — VRAM residency + batched inference + Worker seam      ← production
sandbox       Isolation       — per-unit ephemeral sandbox + brokered credentials     ← production
stream        Verdict Stream  — durable, append-only, idempotent verdict log          ← production
host          Serving Host    — vLLM residency materialized on the box over SSH       ← production
executor      Engine          — composes everything; persists to .kiln/state.json
cli           kiln / kiln-conform binaries
```

Each bounded context is one crate. Every aggregate is a **decider**
(`decide() -> Result<Vec<Event>, &'static str>`, where `Err` is the violated
invariant id); every read-model is a **projector** (an event fold).

---

## Install & build

```bash
cargo build --release        # produces target/release/kiln and kiln-conform
cargo test                   # 97 tests across the workspace
```

## CLI usage

```bash
kiln manifest                       # publish the CapabilityManifest (what can run here)
kiln mode set queue                 # throw the developer switch into QUEUE
kiln admit work-unit.json           # admit a frozen WorkUnit (structural + capability pre-flight)
kiln run                            # drain the queue (in-memory demo path)
kiln serve                          # drain isolated: sandbox + creds + worker + oracle + durable log
kiln status                         # box mode + read-model views
kiln stream                         # print the emitted VerdictEvents
kiln mode set explorer && kiln explore   # run a discovery session (EXPLORER only)
```

A WorkUnit is the **canonical contract JSON** — kebab-case throughout (`unit-ref`,
`spmc-bundle`, `cell-graph`, `acceptance-class` ∈ `auto-commit-if-green` |
`needs-verdict`, `artifact-delivery` ∈ `inline` | `repository`). See
[`examples/workunit-csharp.json`](examples/workunit-csharp.json) (inline) and
[`examples/workunit-csharp-repo.json`](examples/workunit-csharp-repo.json)
(repository), plus [`docs/production-seams.md`](docs/production-seams.md) for the
full `kiln serve` pipeline.

State persists to `.kiln/state.json`; the durable verdict log to
`.kiln/verdicts.jsonl`; per-unit sandboxes under `.kiln/sandboxes/`.

### Pointing `serve` at a real model

`kiln serve` runs each unit's frontier through a **Worker** and gates it with a
**protected Oracle** — both wired by environment variable:

```bash
# 1. an OpenAI-compatible model server on the box (llama-server / vLLM / TGI)
export KILN_OPENAI_BASE_URL=http://127.0.0.1:8080     # → built-in OpenAiWorker
export KILN_OPENAI_MODEL=qwen2.5-coder-7b             # optional
# 2. the protected gate the worker cannot write (ADR-076)
export KILN_ORACLE_CMD='cargo test --quiet'
kiln mode set queue && kiln admit unit.json && kiln serve
```

Worker precedence: a residency materialized by `kiln mode set` → `KILN_OPENAI_BASE_URL`
(HTTP) → `KILN_WORKER_CMD` (shell) → offline `StubWorker`. **Full box setup:
[`docs/running-on-kiln.md`](docs/running-on-kiln.md).**

### Let the switch start the model (vLLM over SSH)

When `KILN_SSH_TARGET` is set, `kiln mode set` *physically materializes* the
residency: it retires any live host, then launches the mode's model as a **vLLM
container** on the box over SSH, polls its `/v1` endpoint, and only serves once it
answers. The switch becomes a real start/stop of VRAM, not a flag — and `kiln
serve` then auto-targets that host.

```bash
export KILN_SSH_TARGET=dev@spark-abcd.local      # → built-in SshVllmHost backend
export KILN_QUEUE_MODEL=qwen2.5-coder-7b         # model vLLM loads in QUEUE
kiln mode set queue                              # launches the container, waits until ready
```

---

## Spec-driven: the `.product/` graph

The implementation is derived from, and continuously checked against, a Product
Framework model:

- **What** — 8 bounded contexts, 13 entities (11 aggregate roots, each a decider),
  events / commands / invariants / read-models.
- **How** — 9 decisions, 10 principles, 12 patterns; the application contract and the
  Rust crate layout.
- **Deciders & projectors** — every aggregate's guarded state machine, proven
  **sound & complete** by simulation against its scenarios.
- **Behavioural conformance (§6.3)** — `kiln-conform` replays the *realised* Rust
  deciders against the spec's scenario oracle. All 11 deciders are conformant.
- **Deliverables** — acceptance criteria wired to named, passing `cargo test`s and
  **computed**-done.

Re-run the gates:

```bash
product domain validate                       # What graph conformant
product how validate                          # How contract conformant
product archetype check kiln-cli             # crate layout matches the tree
CONF="$PWD/target/release/kiln-conform"
product decider conform box-decider --runner "$CONF box-decider"   # §6.3, per decider
product deliverable done deliverable-serving  # computed-done %
```

---

## Status

- 79/79 tests pass · release build green
- 11/11 deciders behaviourally conformant
- 11/11 deliverables computed-done
- domain / how / archetype all conformant

Physical infrastructure (GPU model serving, microVM isolation, the on-box vLLM
container) is implemented behind the `Worker`, `SandboxRuntime`, `CredentialBroker`,
and `ResidencyHost` **trait seams** with working local backends — a real server, a
container runtime, or the SSH/vLLM host drops in without touching the spec.
