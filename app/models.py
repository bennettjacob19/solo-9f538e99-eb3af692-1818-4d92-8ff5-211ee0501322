"""Model artifact loading, validation and NumPy inference."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ModelValidationError(ValueError):
    """The uploaded weights / bias / labels are not a consistent classifier."""


@dataclass
class Model:
    version: str
    weights: np.ndarray  # shape (n_classes, n_features)
    bias: np.ndarray  # shape (n_classes,)
    labels: List[str]  # length n_classes

    @property
    def n_classes(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.weights.shape[1])

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features + self.bias

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        scores = self.score(features)
        idx = int(np.argmax(scores))
        return self.labels[idx], float(scores[idx])


def _as_matrix(value: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be a rectangular numeric array") from exc
    if arr.ndim != 2:
        raise ModelValidationError(f"{name} must be a 2-D matrix, got {arr.ndim}-D")
    if arr.size == 0:
        raise ModelValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ModelValidationError(f"{name} contains NaN or infinite values")
    return arr


def _as_vector(value: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{name} must be a numeric vector") from exc
    if arr.ndim != 1:
        raise ModelValidationError(f"{name} must be a 1-D vector, got {arr.ndim}-D")
    if arr.size == 0:
        raise ModelValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ModelValidationError(f"{name} contains NaN or infinite values")
    return arr


def validate_artifact(payload: Any) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate an uploaded artifact and return normalised arrays.

    Accepted shape for ``weights`` is ``(n_classes, n_features)``.  ``bias``
    must have ``n_classes`` entries and ``labels`` must name every class.
    """
    if not isinstance(payload, dict):
        raise ModelValidationError("request body must be a JSON object")

    missing = [k for k in ("weights", "bias", "labels") if k not in payload]
    if missing:
        raise ModelValidationError(f"missing required field(s): {', '.join(missing)}")

    weights = _as_matrix(payload["weights"], "weights")
    bias = _as_vector(payload["bias"], "bias")

    labels_raw = payload["labels"]
    if not isinstance(labels_raw, list) or not labels_raw:
        raise ModelValidationError("labels must be a non-empty list of strings")
    labels: List[str] = []
    for item in labels_raw:
        if not isinstance(item, str) or not item:
            raise ModelValidationError("every label must be a non-empty string")
        labels.append(item)
    if len(set(labels)) != len(labels):
        raise ModelValidationError("labels must be unique")

    if bias.shape[0] != weights.shape[0]:
        raise ModelValidationError(
            f"bias has {bias.shape[0]} entries but weights has {weights.shape[0]} classes"
        )
    if len(labels) != weights.shape[0]:
        raise ModelValidationError(
            f"labels has {len(labels)} entries but weights has {weights.shape[0]} classes"
        )
    return weights, bias, labels


def save_artifact(path: str, version: str, payload: Dict[str, Any]) -> None:
    weights, bias, labels = validate_artifact(payload)
    doc = {
        "version": version,
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "labels": labels,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, target)


def load_model(path: str, version: Optional[str] = None) -> Model:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError as exc:
        raise ModelValidationError(f"model file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelValidationError(f"model file is not valid JSON: {exc}") from exc
    weights, bias, labels = validate_artifact(doc)
    resolved = version or doc.get("version") or "unknown"
    model = Model(version=resolved, weights=weights, bias=bias, labels=labels)
    if model.n_features == 0:  # pragma: no cover - guarded by validate
        raise ModelValidationError("weights must have at least one feature column")
    return model


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a free TCP port, then release it for the worker to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
