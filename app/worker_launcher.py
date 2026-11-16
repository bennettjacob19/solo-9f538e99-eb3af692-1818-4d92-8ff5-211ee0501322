"""Launcher for a single model worker process.

Run as ``python -m app.worker_launcher ...`` in its own process group.  It
loads the artifact, binds a loopback port, serves until terminated, and writes
the ready handshake only once the HTTP server is actually accepting.  The API
treats *process exit* as the only proof that a slot has been released.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


async def _serve(
    ready_file: str,
    version: str,
    slow_load_ms: int,
    simulate_load_failure: bool,
    app: object,
    extra: dict | None = None,
) -> int:
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # uvicorn sets TCP_NODELAY itself only for sockets it creates; because we
    # hand it a pre-bound socket we must do it here, otherwise keep-alive
    # responses pay a delayed-ACK penalty (~40ms per request).
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        _write_json_atomic(ready_file, {"ok": False, "error": f"bind failed: {exc}"})
        return 4
    sock.listen(256)
    sock.set_inheritable(True)
    port = int(sock.getsockname()[1])

    if slow_load_ms:
        await asyncio.sleep(slow_load_ms / 1000.0)

    if simulate_load_failure:
        _write_json_atomic(
            ready_file,
            {
                "ok": False,
                "error": "simulated load failure (simulate_load_failure=true)",
            },
        )
        sock.close()
        return 5

    config = uvicorn.Config(
        app,
        log_level="warning",
        access_log=False,
        lifespan="off",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.ensure_future(server.serve(sockets=[sock]))

    deadline = time.time() + 10.0
    while not server.started and time.time() < deadline:
        if serve_task.done():
            _write_json_atomic(ready_file, {"ok": False, "error": "worker server failed to start"})
            return 6
        await asyncio.sleep(0.01)

    _write_json_atomic(
        ready_file,
        {
            "ok": True,
            "port": port,
            "pid": os.getpid(),
            "version": version,
            **(extra or {}),
        },
    )
    try:
        await serve_task
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slot_id")
    parser.add_argument("version")
    parser.add_argument("model_path")
    parser.add_argument("control_dir")
    parser.add_argument("ready_file")
    parser.add_argument("host", nargs="?", default="127.0.0.1")
    parser.add_argument("slow_load_ms", nargs="?", type=int, default=0)
    parser.add_argument("simulate_load_failure", nargs="?", type=int, default=0)
    args = parser.parse_args(argv)

    os.makedirs(args.control_dir, exist_ok=True)

    try:
        from app.models import load_model

        started = time.time()
        model = load_model(args.model_path, version=args.version)
        load_seconds = time.time() - started
    except Exception as exc:  # report load failure instead of dying silently
        _write_json_atomic(
            args.ready_file,
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        return 2

    from app.worker import create_worker_app

    app = create_worker_app(args.version, args.slot_id, args.model_path)

    try:
        return asyncio.run(
            _serve(
                ready_file=args.ready_file,
                version=args.version,
                slow_load_ms=max(0, args.slow_load_ms),
                simulate_load_failure=bool(args.simulate_load_failure),
                app=app,
                extra={
                    "n_features": model.n_features,
                    "n_classes": model.n_classes,
                    "load_seconds": round(load_seconds, 3),
                },
            )
        )
    except Exception as exc:  # pragma: no cover - child process
        _write_json_atomic(args.ready_file, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
