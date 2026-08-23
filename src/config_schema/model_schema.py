from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    _target_: str = MISSING
    backbone: Optional[dict[str, Any]] = None
    adapter: Optional[dict[str, Any]] = None
    head: Optional[dict[str, Any]] = None


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model", name="model_schema", node=ModelConfig)
