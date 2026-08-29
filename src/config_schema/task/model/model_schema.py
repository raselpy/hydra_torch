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


def setup_config() -> None:
    cs = ConfigStore.instance()

    cs.store(
        group="task/model",
        name="model_schema",
        node=ModelConfig,
    )

    backbone_schema.setup_config()
    adapter_schema.setup_config()
    head_schema.setup_config()
