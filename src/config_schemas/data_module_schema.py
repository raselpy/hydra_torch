from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DataModuleConfig:
    _target_: str = MISSING


@dataclass
class MNISTDataModuleConfig(DataModuleConfig):
    _target_: str = "src.modular.data.Data"
    data: str = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data", name="data_module_schema", node=MNISTDataModuleConfig)