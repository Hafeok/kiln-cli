# Running on the Spark

How to run real autonomous work on a DGX Spark box (or any Linux machine with a
served model). `spark` is the **executor** — it orchestrates a model server and a
verification gate through two seams. This guide wires both to real infrastructure.

```
   developer ── spark mode set queue ──▶  box residency: QUEUE
        │
   spark admit unit.json  ──▶  queue (binding-homogeneity guard)
        │
   spark serve  ──┬─▶  Worker   = a served model        (SPARK_OPENAI_BASE_URL | SPARK_WORKER_CMD)
                  └─▶  Oracle    = a protected test/gate (SPARK_ORACLE_CMD)
                          │
                          ▼
                  durable verdict log  .spark/verdicts.jsonl
```

---

## 1. Prerequisites

```bash
# build the executor
cargo build --release          # → target/release/spark, target/release/spark-conform
export PATH="$PWD/target/release:$PATH"
```

On the box you also need **one served model** reachable over HTTP, and a
**protected oracle** command (the worker must not be able to write it).

---

## 2. Serve a model on the box

Pick whatever you run on the Spark. `spark` only needs an OpenAI-compatible
`/v1/chat/completions` endpoint.

**llama.cpp (`llama-server`)**

```bash
llama-server \
  -m /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 8        # --parallel N enables batched serving
```

**vLLM**

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --quantization awq --host 127.0.0.1 --port 8080 --api-key spark-local
```

Confirm it answers:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":false}' | head -c 200
```

> **Residency & the switch.** `spark mode set queue` flips the box's *residency
> state* into QUEUE and enforces the one-binding-at-a-time invariants. It does not
> itself load the GGUF — your model server owns VRAM. Keep **one** server resident
> per mode: that is the QUEUE-vs-EXPLORER mutual exclusion in practice.

---

## 3. Point the Worker seam at it

The built-in `OpenAiWorker` talks to the server directly (one resident model serves
every cell — no per-cell process spawn). Selected automatically when
`SPARK_OPENAI_BASE_URL` is set:

```bash
export SPARK_OPENAI_BASE_URL=http://127.0.0.1:8080   # server root, NOT including /v1
export SPARK_OPENAI_MODEL=qwen2.5-coder-7b           # optional; else the unit's binding.model
export SPARK_OPENAI_API_KEY=spark-local              # optional bearer token (vLLM --api-key)
```

`temperature` and `max_tokens` set in a unit's `model_binding.params` are forwarded
to the request.

**Worker precedence** (first match wins):

| Condition | Worker |
|---|---|
| `SPARK_OPENAI_BASE_URL` set | `OpenAiWorker` (HTTP, persistent server) |
| else `SPARK_WORKER_CMD` set | `CommandWorker` (shells out per cell, prompt on stdin) |
| else | `StubWorker` (deterministic, offline) |

> `OpenAiWorker` speaks plain `http://` (on-box localhost needs no TLS). For a
> remote TLS endpoint, implement `Worker` with a TLS client — it drops in without
> any spec change.

---

## 4. Point the Oracle seam at a protected gate

The oracle is the verifier the **worker cannot write** (ADR-076). It runs as a shell
command; non-zero exit = cell rejected. The unit's sandbox workspace is the CWD-ish
target — reference it via `$SANDBOX` in your command if you templatize it, or run a
fixed project gate:

```bash
export SPARK_ORACLE_CMD='cargo test --quiet'        # or: pytest -q, make check, ./gate.sh
```

If unset, `serve` uses a trivially-passing **protected** oracle (`worker_writable:
false`) so the loop runs offline. A `CommandOracle` with `worker_writable: true`
**fails closed** — an oracle the worker can write is never trusted, even if it exits 0.

---

## 5. Run it

```bash
spark mode set queue
spark admit unit.json            # repeat to enqueue more frozen WorkUnits
spark serve                      # isolated drain over the whole queue
```

`serve` prints which worker it selected, then per unit:

```
worker: OpenAI HTTP @ http://127.0.0.1:8080
  accepted   wu-hello  (sandbox provisioned → worker → oracle → logged → torn down)
isolated-drained 1 unit-attempt(s); durable log holds 1 verdict(s) at .spark/verdicts.jsonl
```

Inspect results:

```bash
spark status                     # box mode + read-model views
spark stream                     # emitted VerdictEvents
cat .spark/verdicts.jsonl        # the durable, append-only log (idempotent by bundle_hash)
```

### EXPLORER mode

One large model, serial — for discovery rather than batched units. Swap the resident
server for your big model, then:

```bash
spark mode set explorer
spark explore                    # produces a discovery record (candidate structure, NOT accepted code)
```

---

## 6. What each step enforces

| Step | Invariant / guard |
|---|---|
| `mode set` | `inv-distinct-mode` — a no-op flip is refused; flips are a deliberate human act |
| `admit` | binding-homogeneity — a mixed-binding unit is a decomposition defect, never dispatched |
| sandbox provision | `inv-undeclared-network` — only declared destinations reachable |
| credential lease | `inv-lease-needs-sandbox` — authority bound to the live sandbox |
| batch | `inv-batch-homogeneous` / `inv-batch-empty` — confound-free inference call |
| gate | `inv-oracle-writable` (ADR-076) — gate only against a worker-unwritable oracle |
| emit | `inv-idempotent-append` — at most one verdict per `bundle_hash` |
| teardown | sandbox destroyed, lease revoked — nothing standing survives the unit |

---

## 7. Files & state

| Path | Contents |
|---|---|
| `.spark/state.json` | persisted Engine (queue, views, sequence) |
| `.spark/verdicts.jsonl` | durable, append-only verdict log |
| `.spark/sandboxes/<unit>/` | per-unit workspace (removed at verdict) |

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `worker: offline stub …` printed | neither `SPARK_OPENAI_BASE_URL` nor `SPARK_WORKER_CMD` set |
| `connect 127.0.0.1:8080: …` | model server not up / wrong port |
| `non-200 from …` | bad model name, missing/incorrect API key |
| `only http:// urls supported` | `OpenAiWorker` is `http`-only; use a TLS `Worker` for `https` |
| units `escalated` then `halted` | the oracle rejected; check `SPARK_ORACLE_CMD` runs green by hand |
| `rejected '…': inv-…` on admit | the WorkUnit isn't binding-homogeneous — fix the decomposition |

See [`production-seams.md`](production-seams.md) for the architecture behind each seam.
