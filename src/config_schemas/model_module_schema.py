from typing import Any

from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING
from src.config_schemas import backbone_module_schema


@dataclass
class ModelModuleConfig:
    _target_: str = MISSING
    backbone: backbone_module_schema.BackboneModuleConfig = MISSING


@dataclass
class MNISTModelModuleConfig(ModelModuleConfig):
    _target_: str = "src.modular.model.Net"
    num_classes: int = MISSING


def setup_config() -> None:
    backbone_module_schema.setup_config()
    cs = ConfigStore.instance()
    cs.store(group="model", name="model_module_schema", node=MNISTModelModuleConfig)