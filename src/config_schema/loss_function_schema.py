from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class LossFunctionConfig:
    _target_: str = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/loss_function", name="loss_function_schema", node=LossFunctionConfig)
