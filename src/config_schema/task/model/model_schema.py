from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from config_schema.task.model import adapter_schema, backbone_schema, head_schema


@dataclass
class ModelConfig:
    _target_: str = MISSING
    backbone: Optional[dict[str, Any]] = None
    adapter: Optional[dict[str, Any]] = None
    head: Optional[dict[str, Any]] = None


@dataclass
class SimpleModelschema(ModelConfig):
    _target_: str = "hydra_torch.models.model.SimpleModel"


@dataclass
class CIFAR10ModelSchema(ModelConfig):
    _target_: str = "hydra_torch.models.model.CIFAR10Model"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model", name="CIFAR10ModelSchema", node=CIFAR10ModelSchema)
    cs.store(group="task/model", name="SimpleModelschema", node=SimpleModelschema)

    backbone_schema.setup_config()
    adapter_schema.setup_config()
    head_schema.setup_config()
