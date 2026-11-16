#!/usr/bin/env python3
"""End-to-end demo of the multi-device hot-swap inference service.

Usage::

    python examples/demo.py [BASE_URL]

Covers, against a live API:

0. baseline load;
1. multi-device queueing: FIFO within a device, round-robin across devices,
   one global execution at a time;
2. a successful hot swap;
3. a failed candidate load (old version keeps serving);
4. slow inference with a client timeout, followed by a swap and a third
   upload that is refused until the slow work actually finishes;
5. queued work expiring without being executed.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LABELS = ["pass", "scratch", "dent"]
W1 = [[1.2, -0.4, 0.0, 0.3], [-0.2, 0.9, 0.5, -0.1], [0.1, -0.8, 1.1, 0.2]]
W2 = [[0.3, 1.1, -0.2, 0.1], [1.3, -0.3, 0.1, -0.2], [-0.4, 0.2, 0.9, 0.7]]
FEATURES = [0.7, 0.2, 0.1, 0.5]

client = httpx.Client(base_url=BASE, timeout=60.0)


def show(label, resp):
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"  {label:<42} {resp.status_code}  {json.dumps(body)[:160]}")
    return resp


def slots():
    st = client.get("/admin/status").json()
    layout = [
        (s["slot_id"], s["version"], s["process_state"], s["in_flight"])
        for s in st["slots"]
    ]
    devices = [(d["device_id"], d["pending"], d["executing"]) for d in st["devices"]]
    print(
        f"  status: active={st['active_version']} pending={st['total_pending']} "
        f"in_flight={st['total_in_flight']} slots={layout} devices={devices}"
    )
    return st


def artifact(version, weights, **extra):
    return {"version": version, "weights": weights, "bias": [0.0, 0.0, 0.0], "labels": LABELS, **extra}


def wait_idle(timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/admin/status").json()
        if st["total_pending"] == 0 and st["total_in_flight"] == 0:
            return st
        time.sleep(0.2)
    return client.get("/admin/status").json()


def main() -> int:
    print("\n== 0. baseline load ==")
    show("upload v1", client.post("/admin/models", json=artifact("v1", W1)))
    show("infer", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 1. multi-device queueing (one execution globally) ==")
    order = []

    def ask(device, idx):
        r = client.post(
            "/infer",
            json={
                "features": FEATURES,
                "device_id": device,
                "slow_ms": 350,
                "timeout_ms": 15000,
            },
        )
        order.append((device, r.status_code))

    threads = []
    for i in range(2):
        for device in ("press-1", "press-2", "press-3"):
            t = threading.Thread(target=ask, args=(device, i))
            t.start()
            threads.append(t)
            time.sleep(0.02)
    time.sleep(0.15)
    slots()
    [t.join() for t in threads]
    print(f"  completion order: {[d for d, _ in order]}")
    print("  -> devices alternate; a device's own requests stay in order")
    show("infer without device_id", client.post("/infer", json={"features": FEATURES}))

    print("\n== 2. successful hot swap ==")
    show("upload v2", client.post("/admin/models", json=artifact("v2", W2)))
    show("infer after swap", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 3. failed candidate load leaves v2 active ==")
    show(
        "upload bad-load",
        client.post(
            "/admin/models", json=artifact("bad-load", W1, simulate_load_failure=True)
        ),
    )
    show("infer (still v2)", client.post("/infer", json={"features": FEATURES}))
    show(
        "upload invalid artifact",
        client.post("/admin/models", json=artifact("broken", [[1.0, 2.0]])),
    )
    slots()

    print("\n== 4. slow inference + client timeout ==")
    wait_idle()
    show(
        "slow infer, timeout_ms=150",
        client.post("/infer", json={"features": FEATURES, "slow_ms": 6000, "timeout_ms": 150}),
    )
    st = slots()
    print(f"  in-flight work still running in model process: {st['total_in_flight']}")

    print("\n== 5. swap while the slow call is still running ==")
    show("upload v3", client.post("/admin/models", json=artifact("v3", W1)))
    slots()
    show("upload v4 (third process)", client.post("/admin/models", json=artifact("v4", W2)))
    show("infer (v3 already serving)", client.post("/infer", json={"features": FEATURES}))

    print("\n== 6. queued request expires without executing ==")
    holder = threading.Thread(
        target=lambda: client.post(
            "/infer",
            json={
                "features": FEATURES,
                "device_id": "owner",
                "slow_ms": 1500,
                "timeout_ms": 15000,
            },
        )
    )
    holder.start()
    time.sleep(0.2)
    started = time.monotonic()
    show(
        "queued waiter, timeout_ms=250",
        client.post(
            "/infer",
            json={"features": FEATURES, "device_id": "waiter", "timeout_ms": 250},
        ),
    )
    print(f"  returned after {(time.monotonic() - started) * 1000:.0f}ms")
    slots()
    holder.join()

    print("\n== 7. wait for the slow work to finish; slot is released ==")
    st = wait_idle(timeout=20)
    slots()
    if st["slots_in_use"] <= 1:
        show("upload v4 now", client.post("/admin/models", json=artifact("v4", W2)))
        slots()
    else:
        show(
            "upload v4 still blocked",
            client.post("/admin/models", json=artifact("v4", W2)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
