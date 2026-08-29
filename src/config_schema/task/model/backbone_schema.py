from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    pretrained: bool = False


@dataclass
class ResNet50BackboneSchema(BackboneConfig):
    _target_: str = "hydra_torch.models.backbones.ResNet50"
    pretrained: bool = True


@dataclass
class ResNet18BackboneSchema(BackboneConfig):
    _target_: str = "hydra_torch.models.backbones.ResNet18"
    pretrained: bool = True


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/backbone", name="resnet50_backbone_schema", node=ResNet50BackboneSchema)
    cs.store(group="task/model/backbone", name="resnet18_backbone_schema", node=ResNet18BackboneSchema)
