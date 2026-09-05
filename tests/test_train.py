from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from hydra_torch.scripts import train as train_module


def test_clear_checkpoint_dir_removes_files_and_subdirs(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.ckpt").write_text("fake")
    (checkpoint_dir / "sub").mkdir()
    (checkpoint_dir / "sub" / "nested.txt").write_text("fake")

    train_module.clear_checkpoint_dir(str(checkpoint_dir))

    assert checkpoint_dir.exists()  # the mount-point dir itself must survive
    assert list(checkpoint_dir.iterdir()) == []


def test_clear_checkpoint_dir_creates_dir_if_missing(tmp_path):
    checkpoint_dir = tmp_path / "does_not_exist_yet"
    train_module.clear_checkpoint_dir(str(checkpoint_dir))
    assert checkpoint_dir.exists()


def test_log_run_artifacts_logs_config_tags_and_dvc_lock(tmp_path):
    cfg = OmegaConf.create({"experiment_name": "exp1", "seed": 42})
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"

    dvc_lock_path = tmp_path / "dvc.lock"
    dvc_lock_path.write_text("fake lock content")

    train_module.log_run_artifacts(mlflow_logger, cfg, str(dvc_lock_path))

    mlflow_logger.experiment.log_text.assert_called_once()
    args, _ = mlflow_logger.experiment.log_text.call_args
    assert args[0] == "run123"
    assert args[2] == "resolved_config.yaml"

    mlflow_logger.experiment.set_tag.assert_any_call("run123", "experiment_name", "exp1")
    mlflow_logger.experiment.set_tag.assert_any_call("run123", "seed", 42)
    mlflow_logger.experiment.log_artifact.assert_called_once_with("run123", str(dvc_lock_path))


def test_log_run_artifacts_warns_when_dvc_lock_missing(tmp_path, caplog):
    cfg = OmegaConf.create({"experiment_name": "exp1", "seed": 42})
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"

    missing_path = str(tmp_path / "does_not_exist.lock")

    with caplog.at_level("WARNING"):
        train_module.log_run_artifacts(mlflow_logger, cfg, missing_path)

    mlflow_logger.experiment.log_artifact.assert_not_called()
    assert "dvc.lock not found" in caplog.text


def _make_cfg(tracking_uri):
    return OmegaConf.create({"logger": {"tracking_uri": tracking_uri}})


def test_register_and_promote_champion_skips_for_file_store(caplog):
    cfg = _make_cfg("file:./mlruns")
    mlflow_logger = MagicMock()
    task = MagicMock()
    test_results = [{"test_accuracy": 0.9}]

    with caplog.at_level("WARNING"):
        train_module.register_and_promote_champion(cfg, mlflow_logger, task, test_results)

    assert "Skipping model registration" in caplog.text


def _patch_registration(monkeypatch, run_id="run123"):
    monkeypatch.setattr(train_module.mlflow, "set_tracking_uri", MagicMock())

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

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda run_id=None: _FakeRunCtx(run_id))
    monkeypatch.setattr(train_module.mlflow, "log_param", MagicMock())

    fake_model_info = SimpleNamespace(registered_model_version="2")
    monkeypatch.setattr(train_module.mlflow.pytorch, "log_model", MagicMock(return_value=fake_model_info))
    return fake_model_info


def _make_task():
    return SimpleNamespace(
        model=SimpleNamespace(
            backbone=SimpleNamespace(),
            adapter=SimpleNamespace(),
        )
    )


def test_register_and_promote_champion_promotes_when_better(monkeypatch):
    cfg = _make_cfg("http://localhost:5000")
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"
    task = _make_task()
    test_results = [{"test_accuracy": 0.9}]

    _patch_registration(monkeypatch)

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="old_run")
    fake_client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"test_accuracy": 0.7}))
    monkeypatch.setattr(train_module.mlflow.tracking, "MlflowClient", lambda: fake_client)

    train_module.register_and_promote_champion(cfg, mlflow_logger, task, test_results)

    fake_client.set_registered_model_alias.assert_called_once_with(
        name="hydra_torch_SimpleNamespace", alias="champion", version="2"
    )


def test_register_and_promote_champion_does_not_promote_when_worse(monkeypatch):
    cfg = _make_cfg("http://localhost:5000")
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"
    task = _make_task()
    test_results = [{"test_accuracy": 0.5}]

    _patch_registration(monkeypatch)

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.return_value = SimpleNamespace(run_id="old_run")
    fake_client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={"test_accuracy": 0.7}))
    monkeypatch.setattr(train_module.mlflow.tracking, "MlflowClient", lambda: fake_client)

    train_module.register_and_promote_champion(cfg, mlflow_logger, task, test_results)

    fake_client.set_registered_model_alias.assert_not_called()


def test_register_and_promote_champion_promotes_when_no_prior_champion(monkeypatch):
    cfg = _make_cfg("http://localhost:5000")
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"
    task = _make_task()
    test_results = [{"test_accuracy": 0.6}]

    _patch_registration(monkeypatch)

    fake_client = MagicMock()
    fake_client.get_model_version_by_alias.side_effect = train_module.mlflow.exceptions.RestException(
        {"error_code": "RESOURCE_DOES_NOT_EXIST"}
    )
    monkeypatch.setattr(train_module.mlflow.tracking, "MlflowClient", lambda: fake_client)

    train_module.register_and_promote_champion(cfg, mlflow_logger, task, test_results)

    fake_client.set_registered_model_alias.assert_called_once()


def test_register_and_promote_champion_swallows_unexpected_errors(monkeypatch, caplog):
    cfg = _make_cfg("http://localhost:5000")
    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run123"
    task = _make_task()
    test_results = [{"test_accuracy": 0.6}]

    monkeypatch.setattr(
        train_module.mlflow, "set_tracking_uri", MagicMock(side_effect=RuntimeError("boom"))
    )

    with caplog.at_level("WARNING"):
        train_module.register_and_promote_champion(cfg, mlflow_logger, task, test_results)

    assert "Model registration failed" in caplog.text