from typing import Any

from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class ModelModuleConfig:
    _target_: str = MISSING


@dataclass
class MNISTModelModuleConfig(ModelModuleConfig):
    _target_: str = "src.modular.model.Net"
    num_classes: int = MISSING
    # backbone: Any = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", name="model_module_schema", node=MNISTModelModuleConfig)