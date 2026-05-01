from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from pydantic_core import MISSING
from src.config_schemas import backbone_schema

@dataclass
class ModelConfig:
    _target_: str = MISSING

@dataclass
class SimpleModelConfig(ModelConfig):
    _target_: str = "src.modular.model.Net"
    backbone: backbone_schema.BackboneConfig = MISSING


def setup_config():
    cs = ConfigStore.instance()
    cs.store(group="model", name="model_schema", node=ModelConfig)
