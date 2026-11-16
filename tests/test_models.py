"""Unit tests for artifact validation and NumPy scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.models import (
    Model,
    ModelValidationError,
    load_model,
    save_artifact,
    validate_artifact,
)


def test_validate_accepts_consistent_artifact():
    weights, bias, labels = validate_artifact(
        {
            "weights": [[1.0, 2.0], [3.0, 4.0]],
            "bias": [0.5, -0.5],
            "labels": ["ok", "ng"],
        }
    )
    assert weights.shape == (2, 2)
    assert bias.shape == (2,)
    assert labels == ["ok", "ng"]


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"bias": [0.0], "labels": ["a"]}, "missing required field"),
        ({"weights": [1, 2], "bias": [0.0], "labels": ["a"]}, "2-D"),
        ({"weights": [[1], [2]], "bias": [0.0], "labels": ["a", "b"]}, "bias has 1"),
        ({"weights": [[1], [2]], "bias": [0.0, 0.0], "labels": ["a"]}, "labels has 1"),
        ({"weights": [[1]], "bias": [0.0], "labels": ["a", "a"]}, "unique"),
        ({"weights": [[float("nan")]], "bias": [0.0], "labels": ["a"]}, "finite"),
        ({"weights": [[1]], "bias": [0.0], "labels": ["a", "b"]}, "labels has"),
        ({"weights": [], "bias": [], "labels": []}, "2-D"),
        ({"weights": [[1]], "bias": [0.0], "labels": [1]}, "non-empty string"),
    ],
)
def test_validate_rejects_bad_artifacts(payload, fragment):
    with pytest.raises(ModelValidationError) as exc:
        validate_artifact(payload)
    assert fragment in str(exc.value)


def test_predict_is_argmax_of_linear_scores():
    model = Model(
        version="v1",
        weights=np.array([[1.0, 0.0], [0.0, 2.0]]),
        bias=np.array([0.0, 0.1]),
        labels=["defect-A", "defect-B"],
    )
    label, score = model.predict(np.array([0.2, 0.3]))
    assert label == "defect-B"
    assert score == pytest.approx(0.7)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "m.json"
    save_artifact(str(path), "v9", {"weights": [[1, 2]], "bias": [0.5], "labels": ["only"]})
    model = load_model(str(path))
    assert model.version == "v9"
    assert model.n_features == 2
    assert model.labels == ["only"]
    # persisted document carries the version for bootstrap reloads
    assert json.loads(path.read_text())["version"] == "v9"


def test_load_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ModelValidationError, match="not valid JSON"):
        load_model(str(path))
