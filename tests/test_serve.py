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
    from fastapi.testclient import TestClient

    import hydra_torch.serving.serve as serve_module

    monkeypatch.setattr(serve_module, "_model", None)
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)
    with TestClient(serve_module.app) as client:
        response = client.get("/health")
    assert response.status_code == 503


def test_health_returns_ok_when_model_loaded(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    import hydra_torch.serving.serve as serve_module

    monkeypatch.setattr(serve_module, "_model", MagicMock())
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)  # skip real lifespan loading
    with TestClient(serve_module.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
def test_load_model_loads_and_sets_globals(monkeypatch):
    from unittest.mock import MagicMock

    import hydra_torch.serving.serve as serve_module

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    fake_model = MagicMock()
    fake_datamodule_instance = MagicMock()
    fake_datamodule_instance.transform = "fake_transform"

    monkeypatch.setattr(serve_module.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(serve_module.mlflow.pytorch, "load_model", lambda uri, map_location: fake_model)
    monkeypatch.setattr(
        serve_module, "CIFAR10DataModule", lambda batch_size: fake_datamodule_instance
    )

    serve_module._load_model()

    fake_model.eval.assert_called_once()
    fake_model.to.assert_called_once_with(serve_module.DEVICE)
    assert serve_module._model is fake_model
    assert serve_module._transform == "fake_transform"


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    from fastapi.testclient import TestClient

    import hydra_torch.serving.serve as serve_module

    monkeypatch.setattr(serve_module, "_model", None)
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)
    with TestClient(serve_module.app) as client:
        response = client.post("/predict", files={"file": ("test.png", b"not-a-real-image", "image/png")})
    assert response.status_code == 503


def test_predict_returns_400_for_invalid_image(monkeypatch):
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    import hydra_torch.serving.serve as serve_module

    monkeypatch.setattr(serve_module, "_model", MagicMock())
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)
    with TestClient(serve_module.app) as client:
        response = client.post("/predict", files={"file": ("bad.png", b"not-a-real-image", "image/png")})
    assert response.status_code == 400


def test_predict_returns_prediction_for_valid_image(monkeypatch):
    import io
    from unittest.mock import MagicMock

    import torch
    from fastapi.testclient import TestClient
    from PIL import Image

    import hydra_torch.serving.serve as serve_module

    fake_model = MagicMock(return_value=torch.tensor([[10.0, 1.0] + [0.0] * 8]))
    monkeypatch.setattr(serve_module, "_model", fake_model)
    monkeypatch.setattr(serve_module, "_transform", lambda img: torch.randn(3, 32, 32))
    monkeypatch.setattr(serve_module, "_load_model", lambda: None)

    image = Image.new("RGB", (32, 32))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    with TestClient(serve_module.app) as client:
        response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == serve_module.CIFAR10_CLASSES[0]
    assert len(body["probabilities"]) == 10
