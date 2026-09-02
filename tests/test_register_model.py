from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hydra_torch.serving import register_model as rm


def test_build_model_from_checkpoint_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.ckpt"
    with pytest.raises(FileNotFoundError):
        rm._build_model_from_checkpoint(str(missing), "cpu")


def test_register_raises_without_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        rm.register("checkpoints/best.ckpt", "some_model", "champion")


class _FakeRunInfo:
    def __init__(self, run_id):
        self.run_id = run_id


class _FakeRunCtx:
    def __init__(self, run_id):
        self.info = _FakeRunInfo(run_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_common(monkeypatch, run_id="run123"):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setattr(rm, "_build_model_from_checkpoint", lambda path, device: MagicMock())
    monkeypatch.setattr(rm.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(rm.mlflow, "start_run", lambda run_name=None: _FakeRunCtx(run_id))
    monkeypatch.setattr(rm.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(rm.mlflow.pytorch, "log_model", MagicMock())
    return run_id


def test_register_raises_when_no_version_matches_run(monkeypatch):
    _patch_common(monkeypatch)
    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = []
    monkeypatch.setattr(rm, "MlflowClient", lambda: fake_client)

    with pytest.raises(RuntimeError, match="could not find a version"):
        rm.register("checkpoints/best.ckpt", "some_model", "champion")


def test_register_raises_when_alias_resolves_to_wrong_version(monkeypatch):
    run_id = _patch_common(monkeypatch)
    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [
        SimpleNamespace(run_id=run_id, version="1"),
    ]
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(version="2")
    monkeypatch.setattr(rm, "MlflowClient", lambda: fake_client)

    with pytest.raises(RuntimeError, match="did not take effect"):
        rm.register("checkpoints/best.ckpt", "some_model", "champion")


def test_register_success_sets_alias_to_correct_version(monkeypatch):
    run_id = _patch_common(monkeypatch)
    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [
        SimpleNamespace(run_id=run_id, version="3"),
    ]
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(version="3")
    monkeypatch.setattr(rm, "MlflowClient", lambda: fake_client)

    rm.register("checkpoints/best.ckpt", "some_model", "champion")

    fake_client.set_registered_model_alias.assert_called_once_with(
        name="some_model", alias="champion", version=3
    )