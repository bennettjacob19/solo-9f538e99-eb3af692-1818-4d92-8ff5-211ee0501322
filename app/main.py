"""Edge QC classifier inference service with two-process hot model swapping.

Public surface (see ``/docs``):

* ``POST /infer``            -- queue a feature vector for a device, then classify
* ``POST /admin/models``     -- upload a classifier artifact and hot-swap it in
* ``GET  /admin/versions``   -- version registry
* ``GET  /admin/status``     -- per-version pending/in-flight counts, device queues
* ``GET  /healthz``          -- liveness

The invariants live in ``app/process_manager.py``.  In short: every model
version runs in its own OS process, at most two such processes exist at once,
requests are pinned to the version active at admission, and a retired version
only releases its slot once every inference it accepted has finished inside
its worker -- regardless of whether the client is still waiting.

Device fair queueing lives in ``app/scheduler.py``: each ``device_id`` has a
FIFO queue, non-empty devices rotate round-robin, a device that gets work joins
at the tail of the rotation, and exactly one inference executes globally.
Requests wait up to ``MAX_PENDING_REQUESTS`` (default 32) in the pending queue;
beyond that ``POST /infer`` returns 429.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.models import ModelValidationError, load_model, save_artifact, validate_artifact
from app.process_manager import (
    STATE_DRAINING,
    STATE_IDLE,
    VERSION_ACTIVE,
    VERSION_FAILED,
    VERSION_LOADING,
    VERSION_RETIRED,
    FeatureMismatch,
    NoActiveModel,
    NoCapacity,
    Registry,
    SwapConflict,
    WorkerHandle,
)
from app.scheduler import (
    DEVICE_DEFAULT,
    Job,
    JobDropped,
    QueueFull,
    QueueTimeout,
    ServiceStopping,
    DeviceScheduler,
)

logger = logging.getLogger("qc.service")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "models"
STAGING_DIR = DATA_DIR / "staging"
WORK_DIR = DATA_DIR / "workers"
DEFAULT_TIMEOUT_MS = int(os.environ.get("DEFAULT_INFER_TIMEOUT_MS", "5000"))
MAX_TIMEOUT_MS = 600_000
# Maximum number of admitted-but-not-yet-executing requests across all devices.
MAX_PENDING_REQUESTS = int(os.environ.get("MAX_PENDING_REQUESTS", "32"))
# How long an upload may wait for a draining slot to be released before 429.
SWAP_WAIT_SECONDS = float(os.environ.get("SWAP_WAIT_SECONDS", "3"))
# How long shutdown waits for accepted inference to finish before killing workers.
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "20"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector for the classifier")
    device_id: Optional[str] = Field(
        None,
        max_length=64,
        description=(
            "QC station sharing this service. Requests from one device are "
            "executed first-in-first-out; different devices take turns. "
            "Omitted or empty means the `default` queue."
        ),
    )
    timeout_ms: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_TIMEOUT_MS,
        description=(
            "How long the client is willing to wait from the moment the request "
            "is accepted, in milliseconds. A queued request that expires is "
            "removed without being executed; a dispatched one returns 504 while "
            "the model process keeps computing."
        ),
    )
    timeout_seconds: Optional[float] = Field(
        None,
        gt=0,
        le=MAX_TIMEOUT_MS / 1000.0,
        description=(
            "Same as timeout_ms, expressed in seconds (e.g. 0.2). If both are "
            "provided, timeout_ms wins."
        ),
    )
    slow_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description=(
            "Demo hook: ask the model process to spend this long on the computation "
            "before scoring. Use with a small timeout_ms to demonstrate the 504 / "
            "drain behaviour."
        ),
    )

    @field_validator("features")
    @classmethod
    def _features_present(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("features must contain at least one value")
        return value


class InferResponse(BaseModel):
    version: str
    label: str
    score: float
    inference_id: str
    device_id: str
    elapsed_ms: float
    queue_wait_ms: float


class DeviceQueueView(BaseModel):
    device_id: str
    pending: int
    executing: bool


class SchedulerView(BaseModel):
    max_pending: int
    pending: int
    executing_device: Optional[str]
    admitted_total: int
    executed_total: int
    completed_total: int
    failed_total: int
    queued_timeouts_total: int
    queued_dropped_total: int


class VersionCountView(BaseModel):
    version: str
    state: str
    process_state: str
    slot_id: str
    pid: Optional[int]
    pending: int
    in_flight: int
    accepted: int
    completed: int
    created_at: float
    tested_at: Optional[float]
    activated_at: Optional[float]
    retired_at: Optional[float]
    error: Optional[str]
    attempt: int


class TimeoutResponse(BaseModel):
    detail: Dict[str, Any]


class ModelUpload(BaseModel):
    version: str = Field(..., description="Unique version id, e.g. 'v3'")
    weights: List[List[float]] = Field(..., description="(n_classes x n_features) matrix")
    bias: List[float] = Field(..., description="Bias vector of length n_classes")
    labels: List[str] = Field(..., description="Class label for each weight row")
    test_features: Optional[List[float]] = Field(
        None,
        description="Vector used for the pre-activation smoke test; defaults to all zeros",
    )
    expected_active_version: Optional[str] = Field(
        None,
        description=(
            "Optimistic-concurrency guard: swap only if this is still the active "
            "version. Of concurrent swaps away from the same old version at most "
            "one succeeds."
        ),
    )
    slow_load_ms: int = Field(
        0,
        ge=0,
        le=60_000,
        description="Demo hook: delay worker readiness to observe the old version serving.",
    )
    simulate_load_failure: bool = Field(
        False,
        description="Demo hook: make the worker report a load failure after starting.",
    )


class SlotView(BaseModel):
    slot_id: str
    version: Optional[str]
    process_state: str
    pid: Optional[int]
    pending: int
    in_flight: int
    accepted: int
    completed: int
    alive: Optional[bool]


class StatusView(BaseModel):
    active_version: Optional[str]
    version: Optional[str]
    total_pending: int
    total_in_flight: int
    total_accepted: int
    total_completed: int
    slots_in_use: int
    capacity: int
    slots: List[SlotView]
    versions: List[VersionCountView]
    devices: List[DeviceQueueView]
    scheduler: SchedulerView


class SwapResult(BaseModel):
    version: str
    previous_version: Optional[str]
    slot_id: str
    state: str
    process_state: str
    pid: Optional[int]
    activated_at: float
    test_label: str
    load_seconds: float


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------


class InferenceService:
    def __init__(self) -> None:
        self.manager: Any = None
        self.registry: Registry
        self.scheduler: DeviceScheduler
        self.handles: Dict[str, WorkerHandle] = {}
        self.client: Optional[httpx.AsyncClient] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._closing = False
        self._draining: set[str] = set()
        self._orphaned_waiters: set[asyncio.Future] = set()
        self._swap_lock = asyncio.Lock()  # serialises swaps inside this process
        self._lock_file: Any = None

    # -- lifecycle -------------------------------------------------------
    async def startup(self) -> None:
        for path in (DATA_DIR, MODEL_DIR, STAGING_DIR, WORK_DIR):
            path.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(DATA_DIR / "api.lock", "a+")
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        except OSError as exc:
            raise RuntimeError(
                "another API process already owns this data directory; the "
                "service must run as a single process so the two-slot budget "
                "has a single owner"
            ) from exc
        self.manager = _new_manager()
        self.registry = Registry(self.manager)
        self.scheduler = DeviceScheduler(
            self.registry,
            self._execute_job,
            max_pending=MAX_PENDING_REQUESTS,
        )
        self.scheduler.start()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        self._drain_task = asyncio.create_task(self._drain_loop())
        bootstrap_version = os.environ.get("BOOTSTRAP_VERSION")
        bootstrap_model = os.environ.get("BOOTSTRAP_MODEL")
        if bootstrap_version and bootstrap_model and Path(bootstrap_model).is_file():
            try:
                await self.swap_in_path(
                    version=bootstrap_version,
                    source_path=bootstrap_model,
                    expected_active=None,
                    test_features=None,
                    slow_load_ms=0,
                    simulate_load_failure=False,
                )
                logger.info("bootstrapped model %s", bootstrap_version)
            except Exception:
                logger.exception(
                    "bootstrap model %s failed; service starts without a model",
                    bootstrap_version,
                )

    async def shutdown(self) -> None:
        self._closing = True
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        # Release queued work first: those requests were accepted but never
        # started, so they must not run during shutdown.  A job already
        # executing is shielded and allowed to settle (bounded) by the
        # scheduler before workers are stopped below.
        await self.scheduler.shutdown(grace_seconds=SHUTDOWN_GRACE_SECONDS)
        for handle in list(self.handles.values()):
            await asyncio.to_thread(handle.stop, 5.0)
        self.handles.clear()
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.manager is not None:
            with contextlib.suppress(Exception):
                self.manager.shutdown()
            self.manager = None
        if self._lock_file is not None:
            with contextlib.suppress(Exception):
                self._lock_file.close()
            self._lock_file = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _artifact_path(version: str) -> Path:
        return MODEL_DIR / f"{version}.json"

    def _persist_artifact(self, version: str, source_path: str) -> None:
        """Copy the promoted artifact into the durable per-version location."""
        target = self._artifact_path(version)
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
        with open(source_path, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, target)

    async def _reserve_slot(self, version: str) -> str:
        """Reserve a process slot for a candidate, waiting briefly for a drain.

        A slot that is draining with zero in-flight work is about to be freed
        by the drain loop; waiting a couple of seconds avoids spurious 429s
        without ever exceeding two processes.  If the drain is held up by real
        in-flight work the wait expires and the caller gets 429.
        """
        deadline = time.monotonic() + SWAP_WAIT_SECONDS
        while True:
            try:
                return self.registry.reserve_any_slot(version)
            except SwapConflict as exc:
                # e.g. the version id was registered concurrently, or a retry
                # of a failed id raced a live one.
                raise HTTPException(
                    status_code=409,
                    detail={"error": "version_exists", "message": str(exc)},
                ) from exc
            except NoCapacity as exc:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "capacity_exhausted",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                        headers={"Retry-After": "1"},
                    ) from exc
                await asyncio.sleep(0.05)

    async def _salvage_candidate(
        self, version: str, handle: WorkerHandle, reason: str
    ) -> None:
        """Stop a failed candidate, wait for exit, then free its slot."""
        self.handles.pop(version, None)
        await asyncio.to_thread(handle.stop, 5.0)
        self.registry.fail_candidate(version, reason)
        logger.warning("candidate %s failed: %s", version, reason)

    async def _test_candidate(
        self, version: str, test_features: Optional[List[float]]
    ) -> Dict[str, Any]:
        handle = self.handles[version]
        # Test against the exact artifact file this candidate process loaded,
        # not a shared per-version path (concurrent uploads use staged files).
        artifact = load_model(handle.model_path, version=version)
        features = (
            list(test_features)
            if test_features is not None
            else [0.0] * artifact.n_features
        )
        if len(features) != artifact.n_features:
            raise RuntimeError(
                f"test_features must have {artifact.n_features} values, got {len(features)}"
            )
        assert self.client is not None
        resp = await self.client.post(
            handle.inference_url(),
            json={"features": features},
            timeout=httpx.Timeout(10.0),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"candidate smoke test failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )
        body = resp.json()
        if body.get("version") != version:
            raise RuntimeError("candidate reported a mismatched version")
        if body.get("label") not in artifact.labels:
            raise RuntimeError("candidate returned a label outside the label set")
        return body

    # -- swap ------------------------------------------------------------
    async def swap_in_path(
        self,
        version: str,
        source_path: str,
        expected_active: Optional[str],
        test_features: Optional[List[float]],
        slow_load_ms: int,
        simulate_load_failure: bool,
    ) -> SwapResult:
        """Load a candidate in a fresh process, smoke-test it, then promote it.

        The old version keeps serving until the promotion is a single atomic
        registry operation.  On any failure the candidate is stopped, its slot
        freed, and the old version is left untouched.
        """
        # Snapshot the active version *before* waiting on the in-process swap
        # lock.  ``expected_active`` replaces the snapshot when the caller
        # supplies one.  Competing swaps therefore carry the same old version,
        # and the registry claim (keyed by that version) lets exactly one win.
        observed = (
            expected_active
            if expected_active is not None
            else self.registry.active_version()
        )
        # Fail fast on a known-live version id before taking the swap claim, so
        # the common duplicate-upload case never disturbs the claim table.
        existing = self.registry.version_info(version)
        if existing is not None and existing["state"] != VERSION_FAILED:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "version_exists",
                    "message": f"version {version!r} already exists ({existing['state']})",
                },
            )
        token = uuid.uuid4().hex
        # Claim outside the in-process lock so that a second swap observing the
        # same old version fails immediately instead of queueing behind the
        # first (which may be slow to load) and then failing anyway.
        try:
            old_version = self.registry.begin_swap(observed, token)
        except SwapConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": "swap_conflict", "message": str(exc)},
            ) from exc

        async with self._swap_lock:
            slot_id: Optional[str] = None
            handle: Optional[WorkerHandle] = None
            committed = False
            try:
                # Re-check under the lock: a concurrent request may have
                # registered this version while we waited.
                slot_id = await self._reserve_slot(version)

                handle = WorkerHandle(
                    slot_id=slot_id,
                    version=version,
                    model_path=source_path,
                    work_dir=str(WORK_DIR),
                )
                self.handles[version] = handle
                handle.spawn(
                    slow_load_ms=slow_load_ms,
                    simulate_load_failure=simulate_load_failure,
                )
                self.registry.set_pid(version, handle.pid or -1)

                load_timeout = 30.0 + max(0, slow_load_ms) / 1000.0
                try:
                    ready = await asyncio.to_thread(handle.wait_ready, load_timeout)
                except Exception as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc
                if not ready.get("ok"):
                    reason = str(ready.get("error") or "worker reported load failure")
                    await self._salvage_candidate(version, handle, reason)
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_load_failed",
                            "message": reason,
                            "fallback_version": self.registry.active_version(),
                        },
                    )
                if ready.get("n_features") is not None:
                    self.registry.set_model_shape(
                        version,
                        int(ready["n_features"]),
                        int(ready.get("n_classes", 0)),
                    )

                try:
                    test_body = await self._test_candidate(version, test_features)
                except Exception as exc:
                    await self._salvage_candidate(
                        version, handle, f"smoke test failed: {exc}"
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "candidate_test_failed",
                            "message": str(exc),
                            "fallback_version": self.registry.active_version(),
                        },
                    ) from exc

                self.registry.mark_candidate_tested(version)
                try:
                    self.registry.commit_swap(version, old_version, token)
                except SwapConflict as exc:
                    await self._salvage_candidate(version, handle, str(exc))
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "swap_conflict",
                            "message": str(exc),
                            "active_version": self.registry.active_version(),
                        },
                    ) from exc
                committed = True
                # Keep a durable copy of what is now serving.  Staged uploads
                # are cleaned up by the endpoint; this snapshot is what a
                # future restart can reload.
                try:
                    self._persist_artifact(version, source_path)
                except Exception:
                    logger.exception("failed to persist artifact for %s", version)

                new_info = self.registry.version_info(version)
                assert new_info is not None
                logger.info(
                    "activated model %s in %s (previous=%s, pid=%s)",
                    version,
                    slot_id,
                    old_version,
                    handle.pid,
                )
                return SwapResult(
                    version=version,
                    previous_version=old_version,
                    slot_id=slot_id,
                    state=new_info["state"],
                    process_state=new_info["process_state"],
                    pid=handle.pid,
                    activated_at=float(new_info["activated_at"]),
                    test_label=test_body["label"],
                    load_seconds=float(ready.get("load_seconds", 0.0)),
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("swap of %s failed", version)
                raise HTTPException(
                    status_code=500,
                    detail={"error": "swap_failed", "message": str(exc)},
                ) from exc
            finally:
                if not committed:
                    if handle is not None and version in self.handles:
                        await self._salvage_candidate(version, handle, "swap aborted")
                    elif slot_id is not None:
                        self.registry.fail_candidate(version, "swap aborted")
                self.registry.end_swap(old_version, token)

    # -- draining --------------------------------------------------------
    async def _drain_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(0.05)
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("drain loop error")

    async def drain_once(self) -> None:
        snapshot = self.registry.snapshot()
        for version, info in snapshot["versions"].items():
            handle = self.handles.get(version)
            # Unexpected worker death: a crashed worker cannot be part of the
            # two-process budget any more, but its slot must still be reclaimed
            # so the machine does not appear full forever.
            if handle is not None and not handle.alive():
                self._reap_dead_worker(version, info)

            if info["state"] not in (VERSION_RETIRED, VERSION_FAILED):
                continue
            if info["process_state"] not in (STATE_DRAINING, STATE_IDLE):
                continue
            slot = next(
                (s for s in snapshot["slots"] if s["slot_id"] == info["slot_id"]),
                None,
            )
            if slot is None or slot["version"] != version:
                continue
            if version in self._draining:
                continue
            # The drain gate: accepted work keeps the process alive.  Queued
            # work counts too, because it was bound to this version and must
            # still execute here if it is dispatched in time.
            if self.registry.version_unfinished(version) > 0:
                continue
            self._draining.add(version)
            try:
                if handle is not None:
                    await asyncio.to_thread(handle.stop, 10.0)
                    await asyncio.to_thread(handle.wait_exit, 10.0)
                released = self.registry.release_slot(
                    slot["slot_id"], version, process_exited=True
                )
                if released:
                    self.handles.pop(version, None)
                    logger.info("drained %s; slot %s released", version, slot["slot_id"])
            except Exception:
                logger.exception("failed to drain %s", version)
            finally:
                self._draining.discard(version)

    def _reap_dead_worker(self, version: str, info: Dict[str, Any]) -> None:
        """Reclaim bookkeeping for a worker process that died unexpectedly."""
        self.handles.pop(version, None)
        if info["state"] in (VERSION_ACTIVE, VERSION_LOADING):
            self.registry.settle_dead_worker(version)
            if self.registry.fail_active(version, "worker process exited unexpectedly"):
                logger.error("active model %s died; service has no active model", version)
        elif info["state"] in (VERSION_RETIRED, VERSION_FAILED):
            self.registry.settle_dead_worker(version)
            with contextlib.suppress(Exception):
                self.registry.release_slot(info["slot_id"], version, process_exited=True)

    # -- inference -------------------------------------------------------
    async def infer(
        self,
        features: List[float],
        device_id: Optional[str],
        timeout_ms: int,
        slow_ms: int = 0,
        disconnect: Optional[asyncio.Event] = None,
    ) -> InferResponse:
        """Admit a request to the active version, queue it, and wait for it.

        The clock starts at admission.  While the job is still *pending* its
        expiry or a client disconnect removes it unexecuted; once *dispatched*
        the same events only stop the client waiting -- the shielded executor
        keeps running, keeps its device's turn and keeps the global execution
        slot until the model call actually returns.
        """
        device = (device_id or "").strip() or DEVICE_DEFAULT

        # Capacity and binding are decided in one event-loop step so that no
        # other request can slip between the 429 check and enqueueing.
        if not self.scheduler.has_capacity():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "queue_full",
                    "message": (
                        f"{self.scheduler.pending_count} requests are already "
                        f"waiting (max {self.scheduler.max_pending})"
                    ),
                    "device_id": device,
                    "max_pending": self.scheduler.max_pending,
                },
                headers={"Retry-After": "1"},
            )

        try:
            version, _slot_id = self.registry.admit_work(len(features))
        except FeatureMismatch as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "feature_dimension_mismatch",
                    "message": str(exc),
                    "version": exc.version,
                    "expected_features": exc.expected,
                    "received_features": len(features),
                },
            ) from exc
        except NoActiveModel as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "no_active_model", "message": str(exc)},
            ) from exc

        # Admission binds the request to ``version`` for good.  If the worker
        # is already gone, release the pending count -- it will never run.
        if self.handles.get(version) is None:  # pragma: no cover - invariant
            self.registry.cancel_pending(version)
            raise HTTPException(
                status_code=503,
                detail={"error": "worker_missing", "message": f"worker for {version} is gone"},
            )

        admitted_at = time.monotonic()
        deadline = admitted_at + timeout_ms / 1000.0
        try:
            job = self.scheduler.submit(
                device_id=device,
                version=version,
                features=features,
                slow_ms=slow_ms,
                admitted_at=admitted_at,
                deadline=deadline,
            )
        except QueueFull as exc:  # pragma: no cover - guarded above
            self.registry.cancel_pending(version)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "queue_full",
                    "message": str(exc),
                    "device_id": device,
                    "max_pending": self.scheduler.max_pending,
                },
                headers={"Retry-After": "1"},
            ) from exc

        outcome = await self._await_job(job, disconnect)
        elapsed = (time.monotonic() - admitted_at) * 1000.0
        if outcome["kind"] == "result":
            result = outcome["result"]
            return InferResponse(
                version=version,
                label=result["label"],
                score=float(result["score"]),
                inference_id=job.job_id,
                device_id=device,
                elapsed_ms=round(elapsed, 3),
                queue_wait_ms=job.queue_wait_ms,
            )
        raise outcome["error"]

    async def _await_job(
        self, job: Job, disconnect: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """Wait for ``job`` until it settles, times out, or the client leaves.

        The underlying future is shielded: giving up on waiting never cancels
        work the scheduler has already dispatched.
        """
        wait_task: asyncio.Future = asyncio.ensure_future(asyncio.shield(job.future))
        disc_task: Optional[asyncio.Task] = (
            asyncio.ensure_future(disconnect.wait()) if disconnect is not None else None
        )
        waiters = {wait_task}
        if disc_task is not None:
            waiters.add(disc_task)
        # Give waiting up no later than the admission deadline.  The tiny
        # slack lets a model call that ends right at the deadline win the race
        # so it is reported as a result rather than as a timeout.
        wait_seconds = job.remaining + 0.02
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=wait_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            # Starlette may cancel the handler on disconnect (or on server
            # shutdown).  Queued work must never run in that case; dispatched
            # work is deliberately left alone and shielded.
            if self.scheduler.cancel_pending(job, reason="dropped"):
                logger.info(
                    "dropped queued request %s (device=%s) after handler cancellation",
                    job.job_id,
                    job.device_id,
                )
            raise
        finally:
            if disc_task is not None:
                disc_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await disc_task
            # If we stop waiting while the job is still running (timeout or
            # disconnect), keep a reference to the shielded waiter and drain
            # its eventual result/exception so asyncio does not warn about an
            # unretrieved exception.  The job itself is unaffected.
            if not wait_task.done():
                self._orphaned_waiters.add(wait_task)
                wait_task.add_done_callback(self._discard_orphan)

        if wait_task in done:
            if wait_task.cancelled():  # pragma: no cover - shield invariant
                raise HTTPException(
                    status_code=503,
                    detail={"error": "service_stopping", "message": "request was abandoned"},
                )
            exc = wait_task.exception()
            if exc is None:
                return {"kind": "result", "result": wait_task.result()}
            return {"kind": "error", "error": self._job_error(job, exc)}

        if disc_task is not None and disc_task in done and not wait_task.done():
            # The client went away.  If the job has not been dispatched, drop
            # it so it can never run.  If it has, leave it alone: it keeps its
            # device's turn and the global slot until the model call ends.
            if self.scheduler.cancel_pending(job, reason="dropped"):
                logger.info(
                    "dropped queued request %s (device=%s) after client disconnect",
                    job.job_id,
                    job.device_id,
                )
                raise HTTPException(
                    status_code=499,
                    detail={
                        "error": "client_disconnected",
                        "message": "queued request removed before execution",
                        "device_id": job.device_id,
                        "version": job.version,
                    },
                )
            logger.info(
                "client disconnected from dispatched request %s; it keeps running",
                job.job_id,
            )
            raise HTTPException(
                status_code=499,
                detail={
                    "error": "client_disconnected",
                    "message": "request is already executing and will run to completion",
                    "device_id": job.device_id,
                    "version": job.version,
                },
            )

        # Neither settled nor disconnected within the admission deadline.
        if self.scheduler.cancel_pending(job, reason="timeout"):
            logger.warning(
                "queued request %s (device=%s) expired after %dms; not executed",
                job.job_id,
                job.device_id,
                job.deadline and int((job.deadline - job.admitted_at) * 1000),
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "queue_timeout",
                    "version": job.version,
                    "device_id": job.device_id,
                    "message": (
                        "request timed out while waiting for its turn; it was "
                        "removed and never executed"
                    ),
                },
            )
        # Lost the race against dispatch: the model is already working on it.
        logger.warning(
            "inference on %s for device %s exceeded its deadline while executing; "
            "worker continues",
            job.version,
            job.device_id,
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": "inference_timeout",
                "version": job.version,
                "device_id": job.device_id,
                "message": (
                    "The client stopped waiting, but the model process continues "
                    "the computation; the next request is not dispatched and the "
                    "model's in-flight count is not cleared until it finishes."
                ),
            },
        )

    def _discard_orphan(self, future: asyncio.Future) -> None:
        """Drop an abandoned shielded waiter, retrieving its outcome."""
        self._orphaned_waiters.discard(future)
        if not future.cancelled():
            with contextlib.suppress(Exception):
                future.exception()

    def _job_error(self, job: Job, exc: BaseException) -> HTTPException:
        if isinstance(exc, QueueTimeout):
            return HTTPException(
                status_code=504,
                detail={
                    "error": "queue_timeout",
                    "version": job.version,
                    "device_id": job.device_id,
                    "message": str(exc),
                },
            )
        if isinstance(exc, ServiceStopping):
            return HTTPException(
                status_code=503,
                detail={"error": "service_stopping", "message": str(exc)},
            )
        if isinstance(exc, JobDropped):
            return HTTPException(
                status_code=502,
                detail={
                    "error": "worker_unavailable",
                    "version": job.version,
                    "device_id": job.device_id,
                    "message": str(exc),
                },
            )
        if isinstance(exc, httpx.HTTPError):
            return HTTPException(
                status_code=502,
                detail={
                    "error": "worker_unreachable",
                    "version": job.version,
                    "message": str(exc),
                },
            )
        return HTTPException(
            status_code=502,
            detail={
                "error": "worker_error",
                "version": job.version,
                "message": str(exc),
            },
        )

    async def _execute_job(self, job: Job) -> Dict[str, Any]:
        """Run one dispatched job to completion inside its pinned worker.

        The scheduler calls this only after ``Registry.dispatch_work`` moved
        the job from pending to in-flight.  Completion (success or failure)
        clears the in-flight count, which is what lets the next job run and
        what eventually lets a retired process drain.
        """
        handle = self.handles.get(job.version)
        if handle is None:
            self.registry.complete_work(job.version)
            raise JobDropped(
                f"worker for version {job.version!r} is gone; request not executed"
            )
        payload: Dict[str, Any] = {"features": job.features}
        if job.slow_ms:
            payload["slow_ms"] = job.slow_ms
        try:
            assert self.client is not None
            resp = await self.client.post(handle.inference_url(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"worker {job.version} returned HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
            return resp.json()
        finally:
            # Runs exactly once when the model call actually settles -- never
            # when a client stops waiting.
            self.registry.complete_work(job.version)

    # -- status ----------------------------------------------------------
    def status(self) -> StatusView:
        """Build the status view.  Call from the event loop.

        The device queues live in the event loop; reading them from the
        threadpool could race a reassignment mid-iteration.
        """
        snapshot = self.registry.snapshot()
        slots: List[SlotView] = []
        total_pending = 0
        total_in_flight = 0
        total_accepted = 0
        total_completed = 0
        active = snapshot["active_version"]
        for slot in snapshot["slots"]:
            handle = self.handles.get(slot["version"]) if slot["version"] else None
            alive = handle.alive() if handle is not None else None
            total_pending += int(slot.get("pending", 0))
            total_in_flight += int(slot["in_flight"])
            total_accepted += int(slot["accepted"])
            total_completed += int(slot["completed"])
            slots.append(
                SlotView(
                    slot_id=slot["slot_id"],
                    version=slot["version"],
                    process_state=slot["process_state"],
                    pid=slot["pid"],
                    pending=int(slot.get("pending", 0)),
                    in_flight=int(slot["in_flight"]),
                    accepted=int(slot["accepted"]),
                    completed=int(slot["completed"]),
                    alive=alive,
                )
            )
        versions = []
        for info in sorted(
            snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
        ):
            versions.append(
                VersionCountView(
                    version=info["version"],
                    state=info["state"],
                    process_state=info["process_state"],
                    slot_id=info["slot_id"],
                    pid=info.get("pid"),
                    pending=int(info.get("pending", 0)),
                    in_flight=int(info.get("in_flight", 0)),
                    accepted=int(info.get("accepted", 0)),
                    completed=int(info.get("completed", 0)),
                    created_at=float(info.get("created_at", 0.0)),
                    tested_at=info.get("tested_at"),
                    activated_at=info.get("activated_at"),
                    retired_at=info.get("retired_at"),
                    error=info.get("error"),
                    attempt=int(info.get("attempt", 1)),
                )
            )
        devices = [
            DeviceQueueView(
                device_id=d["device_id"],
                pending=d["pending"],
                executing=d["executing"],
            )
            for d in self.scheduler.snapshot()
        ]
        return StatusView(
            active_version=active,
            version=active,
            total_pending=total_pending,
            total_in_flight=total_in_flight,
            total_accepted=total_accepted,
            total_completed=total_completed,
            slots_in_use=snapshot["slots_in_use"],
            capacity=snapshot["capacity"],
            slots=slots,
            versions=versions,
            devices=devices,
            scheduler=SchedulerView(**self.scheduler.stats()),
        )


def _new_manager() -> Any:
    """Create the registry manager, preferring ``fork`` on POSIX.

    The manager server is started once during startup, before the API handles
    requests; using ``fork`` avoids re-importing the uvicorn entry point in a
    spawned child.  On platforms without ``fork`` the default context is used.
    """
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - Windows / spawn-only platforms
        ctx = multiprocessing.get_context()
    return ctx.Manager()


service = InferenceService()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


app = FastAPI(
    title="Edge QC Classifier Inference Service",
    version="1.0.0",
    description=(
        "Linear classifier inference with live model replacement on an edge QC "
        "machine that can host at most **two model processes**.\n\n"
        "* `POST /admin/models` uploads `weights`, `bias` and `labels`; the new "
        "version is loaded in a free process, smoke-tested, then promoted. The "
        "old version keeps serving during load and keeps serving if the new one "
        "fails.\n"
        "* `POST /infer` accepts an optional `device_id` (a QC station). Each "
        "device has a FIFO queue; non-empty devices are served round-robin and "
        "a device joins at the tail. **One inference executes at a time** "
        "globally. Omitted `device_id` means the `default` queue. Up to "
        "`MAX_PENDING_REQUESTS` (32) requests may wait; when full, `POST /infer` "
        "returns 429.\n"
        "* Requests are pinned to the version active when *accepted*. Queued "
        "requests do not count as in-flight but still keep that version's "
        "process alive, so a replacement loaded afterwards does not affect "
        "them. `timeout_ms` (or `timeout_seconds`) counts from acceptance: a "
        "queued request that expires is removed unexecuted (504); a dispatched "
        "one returns 504 while the worker keeps working, keeps its queue turn "
        "and keeps its in-flight count.\n"
        "* `GET /admin/status` shows per-version `pending` and `in_flight` "
        "counts, the device queues, and each process slot's state (`loading` / "
        "`active` / `draining` / `idle`).\n\n"
        "**Demos**\n"
        "- Multi-device queueing: send several `POST /infer` calls with different "
        "`device_id`s and a long `slow_ms`; the label order shows round-robin "
        "service. `/admin/status` shows each device's pending count and the "
        "single executing device.\n"
        "- Queue overflow: send more than 32 concurrent slow requests -> 429.\n"
        "- Slow inference + swap: `POST /infer` with `slow_ms: 3000`, "
        "`timeout_ms: 200`; you get 504 and the version stays `in_flight: 1`. "
        "Upload another version: it becomes active while the old one drains. A "
        "third upload gets 429 until the slow work actually finishes.\n"
        "- Load failure: upload with `simulate_load_failure: true` (or malformed "
        "weights). You get 502 and the old version remains active."
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
def healthz() -> Dict[str, Any]:
    active = service.registry.active_version()
    return {"ok": True, "active_version": active}


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        429: {
            "description": (
                "Pending queue full (MAX_PENDING_REQUESTS, default 32); retry later"
            )
        },
        504: {
            "model": TimeoutResponse,
            "description": (
                "Deadline (measured from admission) expired. Queued: removed "
                "unexecuted. Dispatched: 504 returned, worker keeps computing "
                "and holds its slot."
            ),
        },
        503: {"description": "No active model, or service shutting down"},
        502: {"description": "Worker unreachable or errored"},
    },
    tags=["inference"],
    summary="Queue a feature vector for a device and classify it",
)
async def infer(req: InferRequest, request: Request) -> Any:
    if req.timeout_ms is not None:
        timeout_ms = req.timeout_ms
    elif req.timeout_seconds is not None:
        timeout_ms = max(1, int(round(req.timeout_seconds * 1000)))
    else:
        timeout_ms = DEFAULT_TIMEOUT_MS

    # Starlette cancels the handler task when the client disconnects.  A
    # request that is merely queued must be removed in that case, so watch for
    # the disconnect explicitly; a dispatched request is shielded instead.
    disconnected = asyncio.Event()

    async def _watch_disconnect() -> None:
        while True:
            if await request.is_disconnected():
                disconnected.set()
                return
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(_watch_disconnect())
    try:
        return await service.infer(
            req.features, req.device_id, timeout_ms, req.slow_ms, disconnected
        )
    except HTTPException as exc:
        if exc.status_code == 499:
            # FastAPI would otherwise try to serialise a 499 response to a
            # client that has already gone away.  Close quietly instead.
            return JSONResponse(status_code=204, content=None)
        raise
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watcher


@app.post(
    "/admin/models",
    response_model=SwapResult,
    status_code=201,
    responses={
        409: {"description": "Version exists or swap conflict / stale expectation"},
        429: {"description": "Both model process slots are still occupied"},
        502: {"description": "Candidate load or smoke test failed; old version kept"},
        422: {"description": "Artifact is structurally invalid"},
    },
    tags=["admin"],
    summary="Upload a classifier and hot-swap it in",
)
async def upload_model(payload: ModelUpload) -> SwapResult:
    if not VERSION_RE.match(payload.version):
        raise HTTPException(
            status_code=422,
            detail="version must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        )
    try:
        validate_artifact(payload.model_dump())
    except ModelValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_artifact", "message": str(exc)},
        ) from exc

    target = STAGING_DIR / f"{payload.version}-{uuid.uuid4().hex[:12]}.json"
    try:
        save_artifact(str(target), payload.version, payload.model_dump())
    except (OSError, ModelValidationError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "artifact_write_failed", "message": str(exc)},
        ) from exc

    try:
        result = await service.swap_in_path(
            version=payload.version,
            source_path=str(target),
            expected_active=payload.expected_active_version,
            test_features=payload.test_features,
            slow_load_ms=payload.slow_load_ms,
            simulate_load_failure=payload.simulate_load_failure,
        )
    finally:
        # The worker process holds its own loaded copy in memory; the staging
        # file is only needed for the load.  Published artifacts live under
        # ``models/<version>.json`` and are never overwritten by an upload.
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
    return result


@app.get("/admin/versions", tags=["admin"], summary="Version registry")
def versions() -> Dict[str, Any]:
    snapshot = service.registry.snapshot()
    return {
        "active_version": snapshot["active_version"],
        "capacity": snapshot["capacity"],
        "slots_in_use": snapshot["slots_in_use"],
        "versions": sorted(
            snapshot["versions"].values(), key=lambda v: v.get("created_at", 0)
        ),
    }


@app.get(
    "/admin/status",
    response_model=StatusView,
    tags=["admin"],
    summary="Per-version pending/in-flight counts, device queues, process states",
)
async def status() -> StatusView:
    return service.status()
