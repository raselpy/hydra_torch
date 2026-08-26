from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    pretrained: bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/backbone", name="backbone_schema", node=BackboneConfig)
