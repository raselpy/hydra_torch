from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BackboneConfig:
    _target_: str = MISSING


@dataclass
class ResNet18BackboneConfig(BackboneConfig):
    _target_: str = "backbones.ResNet18"
    pretrained: bool = True



def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/backbone", name="backbone_schema", node=ResNet18BackboneConfig)