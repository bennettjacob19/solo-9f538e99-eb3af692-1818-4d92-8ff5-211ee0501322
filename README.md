# Edge QC Classifier Inference Service

A FastAPI + NumPy inference service for an edge quality-control machine with
two hard constraints:

1. **the machine can host at most two model processes**, and models are
   replaced online without dropping accepted inference;
2. **several QC stations share the compute**, so only one inference runs at a
   time globally and one device cannot monopolise the service.

```
POST /infer           classify a feature vector (optional device_id / timeout)
POST /admin/models    upload weights/bias/labels and hot-swap the model in
GET  /admin/versions  version registry
GET  /admin/status    per-version pending/in-flight counts, device queues, slots
GET  /healthz         liveness
GET  /docs            Swagger UI (interactive; shows every field)
```

## Running it

```bash
docker compose up --build
# API + Swagger UI:  http://localhost:8000/docs
# quick demo:        python examples/demo.py http://localhost:8000
```

Or without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_DIR=./data uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Tests (they start real API and model processes):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Model artifact

Each version is a linear classifier. 120 draws a class per row: `score =
weights @ features + bias`, and the highest-scoring label wins.

```json
{
  "version": "v2",
  "weights": [[1.2, -0.4, 0.0, 0.3],
              [-0.2, 0.9, 0.5, -0.1],
              [0.1, -0.8, 1.1, 0.2]],
  "bias": [-0.10, 0.05, -0.20],
  "labels": ["pass", "scratch", "dent"]
}
```

`weights` must be `n_classes x n_features`, `bias` must have `n_classes`
entries, and `labels` must be `n_classes` unique non-empty strings. Anything
else is rejected with 422 before a process is started.

`POST /admin/models` also accepts these optional fields:

| field | purpose |
| --- | --- |
| `test_features` | vector for the pre-activation smoke test (defaults to zeros) |
| `expected_active_version` | optimistic guard; 409 if the active version changed |
| `slow_load_ms` | demo: delay worker readiness to watch the old version serve |
| `simulate_load_failure` | demo: make the new worker report a load failure |

## Device scheduling

`POST /infer` accepts an optional `device_id` (a QC station). The scheduler in
`app/scheduler.py` runs in the API process:

```json
{"features": [0.7, 0.2, 0.1, 0.5], "device_id": "press-3", "timeout_ms": 5000}
```

| rule | behaviour |
| --- | --- |
| missing/blank `device_id` | goes to the `default` queue |
| ordering inside a device | strict FIFO |
| ordering across devices | round-robin over non-empty devices |
| a device that receives work | joins the rotation at its **tail**; it rejoins at the tail after each turn while it still has work |
| concurrency | **exactly one inference executes globally**; the next item is dispatched only when the previous model call has actually ended (success or error) |
| queue limit | at most **32** requests may be *pending*; a full queue returns **429** with `Retry-After` |
| load smoke test | `POST /admin/models` test inference talks to the candidate worker directly, so it does **not** consume a device queue slot or the global execution turn |

Admission binds a request to the **active model version immediately**, before
it is queued. A request waiting behind other devices keeps using that version
even if a new version is promoted meanwhile; the registry's *pending* count
keeps the old process alive until the queued work either runs or expires.

Queue contents live only in the API process, so a restart starts with an empty
queue.

### Timeouts and disconnects

The clock starts at admission (when the request is accepted and bound to a
version), not when it starts executing.

| state | deadline expires | client disconnects |
| --- | --- | --- |
| **pending** (queued) | removed from its queue, **never executed**, 504 `queue_timeout`; the pending count is released | same: removed, never executed |
| **dispatched** (executing) | 504 `inference_timeout`; the shielded model call keeps running, keeps its device's turn and its in-flight count | request runs to completion; the slot is **not** released early |

Only when the model call actually returns (or raises) does the in-flight
count drop and the next item get dispatched. This is what guarantees a second
device cannot start while the first model process is still busy, and that a
retired process is never stopped mid-computation.

### Status

`GET /admin/status` reports, **per version**, `pending` (admitted, queued) and
`in_flight` (actually executing), plus lifetime `accepted`/`completed` counts
(these survive the version's slot being reused). It also lists each device
queue (`devices`) and scheduler counters
(`scheduler`). Queued work and slow inference do not block swaps or this
endpoint.

## The guarantees and how they are enforced

Each version runs in **its own OS process** (`python -m app.worker_launcher`,
its own process group, private loopback HTTP port). The API process is the only
owner of lifecycle decisions and keeps the registry in a
`multiprocessing.Manager`, so HTTP concurrency cannot race the accounting.
Two fixed slots, `slot-0` and `slot-1`, are the two-process budget.

```
(vacant) --reserve--> loading --commit--> active --retire--> draining --exit--> idle
                           └── load/test failed ──────────────> failed (slot freed)
```

* **At most two model processes.** A slot is reserved only when it is `idle`.
  It becomes `idle` only after the worker's OS process has exited *and* the
  slot's pending and in-flight counters are zero. A `loading` candidate and
  the `active` version both occupy slots, so a third load gets 429 while a
  swap is in progress. The API refuses to start twice against the same data
  directory (file lock), so there is only ever one budget owner.
* **The old version serves during load.** Promotion is one atomic registry
  operation (`commit_swap`) performed only after the candidate answers a
  smoke-test inference. Until then traffic goes to the old process. A failed
  load or smoke test stops the candidate, frees its slot, and returns 502 with
  `fallback_version` = the still-active old version.
* **Accepted inference is pinned to its version.** `admit_work` reads the
  active version *and* increments that slot's pending counter in one lock.
  The returned version is used for the whole request, including while it sits
  in the device queue. A swap that commits afterwards cannot move it.
* **Queued work holds the process too.** The drain gate waits for
  `pending + in_flight == 0`. A queued request keeps its bound version's
  process alive, so after a swap the old version remains `draining` until its
  queue has run or expired.
* **Timeouts do not end dispatched work.** `timeout_ms` (or its seconds alias
  `timeout_seconds`, or the API's default) bounds only how long the caller
  waits. On expiry while executing the client gets 504, but the worker call
  runs shielded from the request timeout/cancellation, and the in-flight
  counter is decremented when the worker actually returns. A disconnected
  client behaves the same way.
* **Queued work that expires never runs.** Removal from the queue and the
  registry pending decrement happen together; the scheduler never dispatches
  a job past its admission deadline.
* **The old process exits only after real work ends.** The drain loop stops and
  reaps a `draining` process only when its unfinished count is zero. Just after
  a 504 you will see the old version `draining` with `in_flight: 1`; the slot
  is not reusable yet.
* **A full machine refuses the third version, without touching the current
  one.** While both slots are occupied, uploads get 429 (after a short bounded
  wait for a drain that is already finishing). The active version is never
  stopped to make room.
* **Concurrent replacement of the same old version: at most one wins.** A swap
  claims its old version in the registry before starting a process. A second
  swap from the same old version gets 409 immediately and never spawns a
  process. A stale `expected_active_version` also yields 409.
* **Crashed workers are reclaimed.** If a worker dies unexpectedly, its slot
  is released and, if it was active, the service reports 503 (no active model)
  rather than routing to a dead process.

## Demo scenarios

`python examples/demo.py <base-url>` runs all of these end to end.

1. **Multi-device round-robin.** Five requests each from `press-1`,
   `press-2` and `press-3` with `slow_ms`; the completion order shows one
   device at a time and round-robin between non-empty devices (not FIFO by
   arrival across devices).
2. **Normal swap.** Upload `examples/models/v1.json`, infer, upload `v2`,
   infer. `/admin/status` shows the previous version `draining` then `idle`.
3. **Load failure.** Upload with `"simulate_load_failure": true` (or send an
   inconsistent artifact). You get 502/422; the old version keeps answering,
   and the candidate's slot is immediately free again.
4. **Slow inference + timeout + capacity.** `POST /infer` with
   `{"features": [...], "slow_ms": 6000, "timeout_ms": 150}`. You get 504.
   `/admin/status` still shows `in_flight: 1`. Upload a new version — it
   becomes active and the slow version goes `draining`. A third upload returns
   429 while the slow computation still runs. After it finishes the slot is
   released and the upload succeeds.
5. **Queued expiry.** Hold the executor with a slow request, then send another
   request from a different device with a short `timeout_ms`. It returns 504
   `queue_timeout` without ever running; the status `pending` count drops.

## Observations from the smoke runs

- Single request latency on loopback: ~5–7 ms end to end (worker computation
  itself is microseconds; this includes HTTP round trips).
- After a 504, the drain gate held the slot for the full remaining computation
  (verified: 429 until the slow request finished, then `idle`).
- With 3 concurrent requests from one device at `slow_ms=400`, the three
  completions took ~370/670/980 ms: strictly serialized, FIFO.
- Round-robin with devices A/B/C × 2 items each at `slow_ms=400` produced
  queue waits of roughly 0/400/800 for the first round and 1200/1600/2000 for
  the second — alternating devices, not draining one device first.
- A burst of 45 concurrent requests produced `429`s once 32 were pending, and
  the queue drained to zero afterwards.
- SIGTERM with an in-flight request: the request completed with 200, queued
  work was released unexecuted, the API exited cleanly, and no worker
  processes remained.

## Layout

```
app/
  main.py             API, swap orchestration, drain loop, status
  process_manager.py  shared registry/state machine + worker process handles
  scheduler.py        per-device FIFO queues, round-robin, single executor
  worker.py           per-version HTTP server (one process per version)
  worker_launcher.py  process entry point: load artifact, bind, serve, handshake
  models.py           artifact validation, NumPy scoring
tests/
  test_models.py          artifact validation + scoring
  test_registry.py        state machine, pending/in-flight, claims, drain gate
  test_scheduler.py       round-robin, FIFO, capacity, cancellation
  test_api_integration.py real API + real worker processes
examples/
  demo.py, models/*.json
```

## Configuration

| env var | default | meaning |
| --- | --- | --- |
| `DATA_DIR` | `/data` | models, staging, worker control dirs, instance lock |
| `DEFAULT_INFER_TIMEOUT_MS` | `5000` | client wait when `timeout_ms` is omitted |
| `MAX_PENDING_REQUESTS` | `32` | maximum admitted-but-unexecuted requests across all devices; beyond it `/infer` returns 429 |
| `SWAP_WAIT_SECONDS` | `3` | how long an upload waits for a draining slot before 429 |
| `SHUTDOWN_GRACE_SECONDS` | `20` | how long shutdown waits for in-flight work |
| `BOOTSTRAP_VERSION` / `BOOTSTRAP_MODEL` | unset | optionally load one artifact at startup |
| `LOG_LEVEL` | `INFO` | logging |
