# tests/test_config_schema.py
import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# Load-bearing side-effecting import — registers Hydra ConfigStore schemas.
# Without this, hydra.compose() below cannot resolve any *_schema default.
from src.config_schema import setup_config  # noqa: F401

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


@pytest.mark.parametrize(
    "task,data_module,trainer",
    [
        ("cifar10_classification", "cifar10", "cpu"),
        ("mnist_classification", "mnist", "cpu"),
    ],
)
def test_config_composes_and_model_instantiates(task, data_module, trainer):
    """Composes the full config for a task/data_module/trainer combination and
    instantiates the model. Does NOT run training or download data — this is
    a cheap, fast check that every _target_ actually resolves to real code
    and every schema is properly registered."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"task={task}", f"data_module={data_module}", f"trainer={trainer}"],
        )
        model = instantiate(cfg.task.model)
        assert model is not None

        loss_function = instantiate(cfg.task.loss_function)
        assert loss_function is not None
