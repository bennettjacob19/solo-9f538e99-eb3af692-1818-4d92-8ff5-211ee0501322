"""Per-device fair queueing for inference, with at most one global execution.

A QC line has several stations ("devices") sharing one inference service.  A
single station must not monopolise the compute, so requests are not executed
on arrival:

* each device has its own FIFO queue;
* non-empty devices are served round-robin;
* a device joins the rotation at the **tail** the first time it has work, and
  rejoins at the tail after each turn while it still has work;
* only **one** inference runs at a time globally.  The next item is dispatched
  only when the previous one has actually finished inside the model process
  (success *or* error), never when the client merely stopped waiting.

Admission is binding: ``InferenceService.infer`` calls
``Registry.admit_work`` (pinning the active version) and only then hands the
job to this scheduler.  A queued job is *pending*, not in-flight: it keeps its
model process alive through the registry's pending counter, but the process is
not "busy" with it.  The scheduler calls ``Registry.dispatch_work`` (an
atomic pending -> in-flight move) immediately before handing the job to the
executor.

Timeout semantics (all deadlines are measured from admission):

* a **pending** job that reaches its deadline, or whose client disconnects, is
  removed from its queue and never executed.  The registry pending count is
  released, which may let a retiring process drain;
* a **dispatched** job that reaches its deadline fails the request with 504,
  but the shielded executor keeps running.  Its in-flight count, its device's
  turn, and the global execution slot are released only when the model call
  actually returns.  A disconnect at this point changes nothing.

The queue lives in the API process (the sole owner of the model budget), so a
restart starts with an empty queue.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set

from app.process_manager import Registry

DEVICE_DEFAULT = "default"

JOB_PENDING = "pending"
JOB_DISPATCHED = "dispatched"
JOB_DONE = "done"
JOB_CANCELLED = "cancelled"


class QueueFull(RuntimeError):
    """The pending queue already holds its maximum number of requests."""


class QueueTimeout(RuntimeError):
    """The job expired while still queued and was removed unexecuted."""


class JobDropped(RuntimeError):
    """The job's bound version/worker disappeared before it could run."""


class ServiceStopping(RuntimeError):
    """The service is shutting down; queued work is released unexecuted."""


class Job:
    __slots__ = (
        "job_id",
        "device_id",
        "version",
        "features",
        "slow_ms",
        "admitted_at",
        "deadline",
        "dispatched_at",
        "finished_at",
        "state",
        "future",
    )

    def __init__(
        self,
        device_id: str,
        version: str,
        features: List[float],
        slow_ms: int,
        admitted_at: float,
        deadline: float,
    ) -> None:
        self.job_id = uuid.uuid4().hex
        self.device_id = device_id
        self.version = version
        self.features = features
        self.slow_ms = slow_ms
        self.admitted_at = admitted_at
        self.deadline = deadline
        self.dispatched_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.state = JOB_PENDING
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        # Requests that time out after dispatch stop awaiting this future.
        # Retrieve any exception here so an unobserved error is not logged as
        # "Future exception was never retrieved"; awaiters still receive it.
        self.future.add_done_callback(_consume_exception)

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def queue_wait_ms(self) -> float:
        if self.dispatched_at is None:
            return 0.0
        return round((self.dispatched_at - self.admitted_at) * 1000.0, 3)


def _consume_exception(future: asyncio.Future) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except Exception:  # pragma: no cover - defensive
        pass


class DeviceScheduler:
    """Round-robin FIFO across devices with a single global executor."""

    def __init__(
        self,
        registry: Registry,
        executor: Callable[[Job], Awaitable[Dict[str, Any]]],
        max_pending: int = 32,
    ) -> None:
        self.registry = registry
        self._executor = executor
        self.max_pending = max(1, int(max_pending))
        self._queues: Dict[str, Deque[Job]] = {}
        self._rotation: Deque[str] = deque()
        self._rotating: Set[str] = set()
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._current: Optional[Job] = None
        self._started_at = time.time()
        self.admitted_total = 0
        self.executed_total = 0
        self.queued_timeouts_total = 0
        self.queued_dropped_total = 0
        self.completed_total = 0
        self.failed_total = 0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="device-scheduler")

    async def shutdown(self, grace_seconds: float = 20.0) -> None:
        """Release queued work, then let the dispatched job settle if it can."""
        self._stopping = True
        self._wake.set()
        self._fail_all_pending(ServiceStopping("service is shutting down"))
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=grace_seconds)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # -- admission -------------------------------------------------------
    @property
    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def has_capacity(self) -> bool:
        return self.pending_count < self.max_pending

    def submit(
        self,
        device_id: str,
        version: str,
        features: List[float],
        slow_ms: int,
        admitted_at: float,
        deadline: float,
    ) -> Job:
        """Queue a job already admitted to ``version``.

        Must be called in the event loop immediately after
        ``Registry.admit_work`` so that the capacity check, version binding
        and enqueue are one uninterrupted step.
        """
        if not self.has_capacity():
            raise QueueFull(
                f"pending queue is full ({self.pending_count}/{self.max_pending})"
            )
        job = Job(device_id, version, features, slow_ms, admitted_at, deadline)
        queue = self._queues.get(device_id)
        if queue is None:
            queue = deque()
            self._queues[device_id] = queue
        if not queue and device_id not in self._rotating:
            self._rotation.append(device_id)
            self._rotating.add(device_id)
        queue.append(job)
        self.admitted_total += 1
        self._wake.set()
        return job

    def cancel_pending(self, job: Job, reason: str = "dropped") -> bool:
        """Remove a queued job so it can never execute.

        Returns True if the job was still pending (and was removed).  A job
        already dispatched returns False: the spec says dispatched work keeps
        running and keeps its slot even if its client goes away.  ``reason``
        only changes which counter is bumped ("timeout" or "dropped").
        """
        if job.state != JOB_PENDING:
            return False
        # Re-check the queue membership before mutating: this runs on the
        # event loop, same as ``_take_next``, so the check cannot race, but a
        # job may already have been popped as expired by the dispatch loop.
        queue = self._queues.get(job.device_id)
        if queue is None or job not in queue:
            return False
        job.state = JOB_CANCELLED
        self.registry.cancel_pending(job.version)
        queue.remove(job)
        if not queue:
            self._queues.pop(job.device_id, None)
            self._drop_from_rotation(job.device_id)
        if reason == "timeout":
            self.queued_timeouts_total += 1
        else:
            self.queued_dropped_total += 1
        return True

    # -- status ----------------------------------------------------------
    def snapshot(self) -> List[Dict[str, Any]]:
        """Device queues as seen from the event loop (status endpoint)."""
        devices: List[Dict[str, Any]] = []
        name: Optional[str] = None
        if self._current is not None:
            name = self._current.device_id
        rotation = list(self._rotation)

        def order_key(device_id: str) -> int:
            # The executing device first, then devices in rotation order.
            if device_id == name:
                return -1
            return rotation.index(device_id) if device_id in rotation else len(rotation)

        ordered = sorted(self._queues, key=order_key)
        if name is not None and name not in self._queues:
            ordered.insert(0, name)
        for device_id in ordered:
            devices.append(
                {
                    "device_id": device_id,
                    "pending": len(self._queues.get(device_id, ())),
                    "executing": device_id == name,
                }
            )
        return devices

    def stats(self) -> Dict[str, Any]:
        return {
            "max_pending": self.max_pending,
            "pending": self.pending_count,
            "executing_device": self._current.device_id if self._current else None,
            "admitted_total": self.admitted_total,
            "executed_total": self.executed_total,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "queued_timeouts_total": self.queued_timeouts_total,
            "queued_dropped_total": self.queued_dropped_total,
            "started_at": self._started_at,
        }

    # -- dispatch loop ---------------------------------------------------
    async def _run(self) -> None:
        while True:
            if self._stopping:
                return
            job = self._take_next()
            if job is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            try:
                result = await self._executor(job)
            except asyncio.CancelledError:
                # Only shutdown cancels the loop; the executor's own ``finally``
                # has already cleared the in-flight count.  Never leave a
                # waiter hanging on a future that can no longer settle.
                if not job.future.done():
                    job.future.set_exception(ServiceStopping("service is shutting down"))
                raise
            except Exception as exc:  # executor already completed the work
                self._settle_exception(job, exc)
            else:
                self._settle_result(job, result)
            finally:
                self._current = None
                self._wake.set()  # let the loop re-check stopping state

    def _take_next(self) -> Optional[Job]:
        """Pop the next job worth executing, cancelling expired queue entries."""
        now = time.monotonic()
        while self._rotation:
            device_id = self._rotation.popleft()
            self._rotating.discard(device_id)
            queue = self._queues.get(device_id)
            if not queue:
                self._queues.pop(device_id, None)
                continue

            job = queue.popleft()
            if job.state != JOB_PENDING:  # pragma: no cover - queue invariant
                if queue:
                    self._rotation.append(device_id)
                    self._rotating.add(device_id)
                else:
                    self._queues.pop(device_id, None)
                continue

            if job.deadline <= now:
                # Accepted, but its time is up before it ever ran: drop it.
                job.state = JOB_CANCELLED
                self.registry.cancel_pending(job.version)
                self.queued_timeouts_total += 1
                if not job.future.done():
                    job.future.set_exception(
                        QueueTimeout(
                            "request timed out while waiting for its turn; it was "
                            "never executed"
                        )
                    )
                if queue:
                    self._rotation.append(device_id)
                    self._rotating.add(device_id)
                else:
                    self._queues.pop(device_id, None)
                continue

            job.state = JOB_DISPATCHED
            job.dispatched_at = time.monotonic()
            if queue:
                self._rotation.append(device_id)
                self._rotating.add(device_id)
            else:
                self._queues.pop(device_id, None)

            # Pending -> in-flight is atomic: the drain gate must never see
            # this version with work queued but nothing counted.
            if not self.registry.dispatch_work(job.version):
                self.registry.cancel_pending(job.version)
                self._settle_exception(
                    job,
                    JobDropped(
                        f"version {job.version!r} can no longer accept work; the "
                        "request was not executed"
                    ),
                )
                continue
            self.executed_total += 1
            self._current = job
            return job
        return None

    def _settle_result(self, job: Job, result: Dict[str, Any]) -> None:
        job.state = JOB_DONE
        job.finished_at = time.monotonic()
        self.completed_total += 1
        if not job.future.done():
            job.future.set_result(result)

    def _settle_exception(self, job: Job, exc: BaseException) -> None:
        job.state = JOB_DONE
        job.finished_at = time.monotonic()
        self.failed_total += 1
        if not job.future.done():
            job.future.set_exception(exc)

    def _fail_all_pending(self, exc: BaseException) -> None:
        for device_id in list(self._queues):
            queue = self._queues.pop(device_id)
            for job in queue:
                job.state = JOB_CANCELLED
                self.registry.cancel_pending(job.version)
                if not job.future.done():
                    job.future.set_exception(exc)
        self._rotation.clear()
        self._rotating.clear()

    def _drop_from_rotation(self, device_id: str) -> None:
        try:
            self._rotation.remove(device_id)
        except ValueError:
            pass
        self._rotating.discard(device_id)
