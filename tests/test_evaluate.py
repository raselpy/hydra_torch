import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from hydra_torch.scripts import evaluate as evaluate_module

_run = evaluate_module.main.__wrapped__


def test_main_raises_without_checkpoint_path():
    cfg = OmegaConf.create({"data_module": {}, "task": {}, "trainer": {}})
    with pytest.raises(ValueError, match="checkpoint_path"):
        _run(cfg)


def test_main_runs_trainer_test_and_writes_metrics(monkeypatch, tmp_path):
    cfg = OmegaConf.create({
        "data_module": {"_target_": "x"},
        "task": {"_target_": "y"},
        "trainer": {"_target_": "z"},
        "checkpoint_path": "checkpoints/best.ckpt",
    })

    fake_data_module = SimpleNamespace()
    fake_task = SimpleNamespace()
    fake_trainer = MagicMock()
    fake_trainer.test.return_value = [{"test_accuracy": 0.42, "test_loss": 1.23}]

    def fake_instantiate(node, **kwargs):
        if node is cfg.data_module:
            return fake_data_module
        if node is cfg.task:
            return fake_task
        if node is cfg.trainer:
            return fake_trainer
        raise AssertionError(f"Unexpected instantiate call: {node}")

    monkeypatch.setattr(evaluate_module, "instantiate", fake_instantiate)
    monkeypatch.chdir(tmp_path)

    _run(cfg)

    fake_trainer.test.assert_called_once_with(
        fake_task, datamodule=fake_data_module, ckpt_path="checkpoints/best.ckpt"
    )

    written = json.loads((tmp_path / "eval_metrics.json").read_text())
    assert written == {"test_accuracy": 0.42, "test_loss": 1.23}