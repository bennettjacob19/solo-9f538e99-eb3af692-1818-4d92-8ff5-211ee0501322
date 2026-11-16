"""HTTP server run inside each model worker process.

A worker process hosts exactly one model version.  It never talks to the
registry directly: the API process owns lifecycle decisions and addresses the
worker over its private loopback HTTP port.  This keeps the worker free of
shared-state races and makes "the process is still working" observable
independently of what any HTTP client (or the API itself) is doing.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.models import Model, load_model


class InferRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector, length must match the model")
    slow_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description="Artificial delay (demo hook) to exercise slow inference / timeouts",
    )


class InferResponse(BaseModel):
    version: str
    label: str
    score: float
    inference_id: str
    elapsed_ms: float


class StatusResponse(BaseModel):
    version: str
    slot_id: str
    pid: int
    in_flight: int
    accepted: int
    completed: int
    started_at: float
    uptime_seconds: float
    n_features: int
    n_classes: int
    labels: List[str]


def create_worker_app(version: str, slot_id: str, model_path: str) -> FastAPI:
    model: Model = load_model(model_path, version=version)

    app = FastAPI(
        title=f"QC classifier worker {version}",
        description="Internal per-version worker. Not intended to be exposed publicly.",
    )
    started_at = time.time()
    counters = {"in_flight": 0, "accepted": 0, "completed": 0}
    counter_lock = threading.Lock()

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True, "version": version, "pid": os.getpid()}

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        with counter_lock:
            in_flight = counters["in_flight"]
            accepted = counters["accepted"]
            completed = counters["completed"]
        return StatusResponse(
            version=version,
            slot_id=slot_id,
            pid=os.getpid(),
            in_flight=in_flight,
            accepted=accepted,
            completed=completed,
            started_at=started_at,
            uptime_seconds=round(time.time() - started_at, 3),
            n_features=model.n_features,
            n_classes=model.n_classes,
            labels=list(model.labels),
        )

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest) -> InferResponse:  # sync: runs in the threadpool
        features = np.asarray(req.features, dtype=np.float64)
        if features.ndim != 1 or features.shape[0] != model.n_features:
            raise HTTPException(
                status_code=400,
                detail=f"expected {model.n_features} features, got {features.shape[0]}",
            )
        if not np.all(np.isfinite(features)):
            raise HTTPException(status_code=400, detail="features contain NaN or infinite values")

        with counter_lock:
            counters["in_flight"] += 1
            counters["accepted"] += 1
        inference_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            # Deliberate demo delay.  The work is done in a *thread*, so an API
            # timeout or client disconnect does not abort it; the API-side
            # in-flight accounting only clears once this call returns.
            if req.slow_ms:
                time.sleep(req.slow_ms / 1000.0)
            label, score = model.predict(features)
            elapsed = (time.perf_counter() - started) * 1000.0
            return InferResponse(
                version=version,
                label=label,
                score=score,
                inference_id=inference_id,
                elapsed_ms=round(elapsed, 3),
            )
        finally:
            with counter_lock:
                counters["in_flight"] -= 1
                counters["completed"] += 1

    return app
