from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BackboneConfig:
    _target_: str = MISSING


@dataclass
class ResNet18BackboneConfig(BackboneConfig):
    _target_: str = MISSING
    pretrained: bool = True



def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model/architectures/backbone", name="backbone_schema", node=ResNet18BackboneConfig)
