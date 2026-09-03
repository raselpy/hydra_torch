# tests/test_serve.py
"""These tests deliberately avoid triggering the app's lifespan (which
requires a real, reachable MLflow registry server) — no such server exists
in CI. Full end-to-end /predict testing against a real registered model is
a known gap, left for manual verification (see README) — noted here
explicitly rather than silently claiming more coverage than actually exists.
"""

import pytest

from hydra_torch.serving.serve import CIFAR10_CLASSES, MODEL_ALIAS, REGISTERED_MODEL_NAME, _load_model


def test_cifar10_classes_match_canonical_order():
    # torchvision.datasets.CIFAR10's label indices, in order — must match
    # exactly, or predictions will report the wrong class name even when
    # the model itself predicted the correct index.
    assert CIFAR10_CLASSES == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    assert len(CIFAR10_CLASSES) == 10


def test_default_registry_settings():
    assert REGISTERED_MODEL_NAME == "hydra_torch_CIFAR10Model"
    assert MODEL_ALIAS == "champion"


def test_load_model_raises_clear_error_without_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        _load_model()

def test_health_returns_503_when_model_not_loaded(monkeypatch):
    import hydra_torch.serving.serve as serve_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve_module, "_model", None)
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)
    with TestClient(serve_module.app) as client:
        response = client.get("/health")
    assert response.status_code == 503


def test_health_returns_ok_when_model_loaded(monkeypatch):
    import hydra_torch.serving.serve as serve_module
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock

    monkeypatch.setattr(serve_module, "_model", MagicMock())
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)  # skip real lifespan loading
    with TestClient(serve_module.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
