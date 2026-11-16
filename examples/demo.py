#!/usr/bin/env python3
"""End-to-end demo of the hot-swap service, runnable against a live API.

Usage::

    python examples/demo.py [BASE_URL]

Covers the three scenarios reviewers care about:

1. normal inference, then a successful hot swap;
2. a failed candidate load (old version keeps serving);
3. slow inference with a client timeout, followed by a swap and a third
   upload that is refused until the slow work finishes and the slot is freed.
"""

from __future__ import annotations

import json
import sys
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
    print(f"  status: active={st['active_version']} slots={layout}")
    return st


def artifact(version, weights, **extra):
    return {"version": version, "weights": weights, "bias": [0.0, 0.0, 0.0], "labels": LABELS, **extra}


def main() -> int:
    print("\n== 0. baseline load ==")
    show("upload v1", client.post("/admin/models", json=artifact("v1", W1)))
    show("infer", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 1. successful hot swap ==")
    show("upload v2", client.post("/admin/models", json=artifact("v2", W2)))
    show("infer after swap", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 2. failed candidate load leaves v2 active ==")
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

    print("\n== 3. slow inference + client timeout ==")
    time.sleep(1.0)  # let the v1 drain finish and free a slot
    show(
        "slow infer, timeout_ms=150",
        client.post("/infer", json={"features": FEATURES, "slow_ms": 6000, "timeout_ms": 150}),
    )
    st = slots()
    print(f"  in-flight work still running in model process: {st['total_in_flight']}")

    print("\n== 4. swap while the slow call is still running ==")
    show("upload v3", client.post("/admin/models", json=artifact("v3", W1)))
    slots()
    show("upload v4 (third process)", client.post("/admin/models", json=artifact("v4", W2)))
    show("infer (v3 already serving)", client.post("/infer", json={"features": FEATURES}))

    print("\n== 5. wait for the slow work to finish; slot is released ==")
    deadline = time.time() + 15
    while time.time() < deadline:
        st = slots()
        if st["slots_in_use"] == 1:
            break
        time.sleep(0.5)
    show("upload v4 now", client.post("/admin/models", json=artifact("v4", W2)))
    slots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
