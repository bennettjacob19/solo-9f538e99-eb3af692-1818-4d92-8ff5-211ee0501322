"""Unit tests for the device round-robin scheduler.

These use a fake executor and a stub registry, so ordering, the capacity limit
and the pending/dispatched cancellation rules are tested without OS processes.
The integration suite covers the same rules end to end.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.scheduler import (
    DEVICE_DEFAULT,
    DeviceScheduler,
    Job,
    QueueFull,
    QueueTimeout,
    ServiceStopping,
)


class StubRegistry:
    """Counts like the real registry, without multiprocessing."""

    def __init__(self) -> None:
        self.pending: dict[str, int] = {}
        self.in_flight: dict[str, int] = {}
        self.completed: dict[str, int] = {}

    def admit(self, version: str) -> None:
        self.pending[version] = self.pending.get(version, 0) + 1

    def cancel_pending(self, version: str) -> bool:
        if self.pending.get(version, 0) <= 0:
            return False
        self.pending[version] -= 1
        return True

    def dispatch_work(self, version: str) -> bool:
        if self.pending.get(version, 0) <= 0:
            return False
        self.pending[version] -= 1
        self.in_flight[version] = self.in_flight.get(version, 0) + 1
        return True

    def complete_work(self, version: str) -> None:
        self.in_flight[version] = max(0, self.in_flight.get(version, 0) - 1)
        self.completed[version] = self.completed.get(version, 0) + 1


def run(coro):
    return asyncio.run(coro)


def enqueue(scheduler: DeviceScheduler, registry: StubRegistry, device: str, timeout: float = 5.0):
    now = time.monotonic()
    registry.admit("v1")
    return scheduler.submit(device, "v1", [0.0], 0, admitted_at=now, deadline=now + timeout)


async def stop(scheduler: DeviceScheduler) -> None:
    scheduler._stopping = True
    scheduler._wake.set()
    if scheduler._task is not None:
        await asyncio.wait_for(scheduler._task, timeout=2.0)
        scheduler._task = None


def test_round_robin_serves_devices_in_turn_and_appends_new_at_tail():
    order: list[str] = []
    registry = StubRegistry()

    async def executor(job: Job):
        order.append(job.device_id)
        await asyncio.sleep(0.005)
        registry.complete_work(job.version)
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=32)
        scheduler.start()
        jobs = [enqueue(scheduler, registry, d) for d in ("A", "B", "A", "C", "B", "A")]
        await asyncio.gather(*(asyncio.shield(j.future) for j in jobs))
        await stop(scheduler)
        return jobs

    jobs = run(scenario())
    # Rotation: [A,B,C] initially. A keeps its place (more work), B too,
    # C empties; order is therefore A,B,C,A,B,A.
    assert order == ["A", "B", "C", "A", "B", "A"]
    assert all(job.state == "done" for job in jobs)
    assert registry.pending == {"v1": 0}
    assert registry.in_flight == {"v1": 0}
    assert registry.completed == {"v1": 6}


def test_omitted_device_uses_default_queue():
    seen: list[str] = []
    registry = StubRegistry()

    async def executor(job: Job):
        seen.append(job.device_id)
        registry.complete_work(job.version)
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=32)
        scheduler.start()
        job = enqueue(scheduler, registry, DEVICE_DEFAULT)
        await asyncio.shield(job.future)
        await stop(scheduler)

    run(scenario())
    assert seen == [DEVICE_DEFAULT]


def test_queue_full_is_rejected():
    registry = StubRegistry()

    async def executor(job: Job):  # pragma: no cover - never dispatched
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=2)
        # Not started: nothing is dispatched, so both entries stay pending.
        enqueue(scheduler, registry, "A")
        enqueue(scheduler, registry, "B")
        assert scheduler.pending_count == 2
        with pytest.raises(QueueFull):
            enqueue(scheduler, registry, "A")

    run(scenario())


def test_pending_job_expires_unexecuted_and_releases_binding():
    registry = StubRegistry()
    executed: list[str] = []

    async def executor(job: Job):  # pragma: no cover - must not run
        executed.append(job.job_id)
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=32)
        # Queue first, start the loop only after the deadline has passed,
        # exactly like a request that waited behind a slow inference.
        job = enqueue(scheduler, registry, "A", timeout=0.05)
        await asyncio.sleep(0.08)
        scheduler.start()
        await asyncio.sleep(0.05)
        assert job.state == "cancelled"
        with pytest.raises(QueueTimeout):
            await asyncio.shield(job.future)
        await stop(scheduler)

    run(scenario())
    assert executed == []
    assert registry.pending == {"v1": 0}
    assert registry.in_flight.get("v1", 0) == 0
    assert registry.completed == {}


def test_cancel_pending_only_works_before_dispatch():
    registry = StubRegistry()
    release = asyncio.Event()

    async def executor(job: Job):
        await release.wait()
        registry.complete_work(job.version)
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=32)
        scheduler.start()
        first = enqueue(scheduler, registry, "A")
        second = enqueue(scheduler, registry, "B")
        await asyncio.sleep(0.05)
        assert first.state == "dispatched"
        assert scheduler.cancel_pending(first) is False
        assert scheduler.cancel_pending(second) is True
        assert second.state == "cancelled"
        release.set()
        await asyncio.sleep(0.05)
        await stop(scheduler)
        return scheduler

    scheduler = run(scenario())
    assert scheduler.executed_total == 1
    assert registry.pending == {"v1": 0}
    assert registry.in_flight == {"v1": 0}


def test_shutdown_releases_queued_work_instead_of_running_it():
    registry = StubRegistry()
    executed: list[str] = []
    release = asyncio.Event()

    async def executor(job: Job):
        executed.append(job.job_id)
        await release.wait()
        registry.complete_work(job.version)
        return {"label": "x"}

    async def scenario():
        scheduler = DeviceScheduler(registry, executor, max_pending=32)
        scheduler.start()
        jobs = [enqueue(scheduler, registry, d) for d in ("A", "B", "C")]
        await asyncio.sleep(0.05)
        release.set()  # let the dispatched one finish
        await scheduler.shutdown(grace_seconds=1.0)
        return jobs

    jobs = run(scenario())
    assert len(executed) == 1
    assert jobs[1].state == "cancelled"
    assert jobs[2].state == "cancelled"
    with pytest.raises(ServiceStopping):
        jobs[1].future.result()
    assert registry.pending == {"v1": 0}
    assert registry.in_flight == {"v1": 0}
