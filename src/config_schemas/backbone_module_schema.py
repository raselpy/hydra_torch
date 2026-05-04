from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BackboneModuleConfig:
    _target_: str = MISSING


@dataclass
class MNISTBackModuleConfig(BackboneModuleConfig):
    _target_: str = "src.modular.backbone.ResNet18"
    src.modular.backbone.ResNet18


def setup_config():
    cs = ConfigStore.instance()
    cs.store(
        group="backbone",
        name="backbone_module_schema",
        node=MNISTBackModuleConfig
    )