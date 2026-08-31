from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = MISSING
    drop_last: bool = MISSING
    data_dir: str = MISSING


@dataclass
class MNISTDataModuleSchema(DataModuleConfig):
    _target_: str = "hydra_torch.data.data_modules.MNISTDataModule"
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class CIFAR10DataModuleSchema(DataModuleConfig):
    _target_: str = "hydra_torch.data.data_modules.CIFAR10DataModule"

    pin_memory: bool = True
    drop_last: bool = True


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data_module", name="mnist_data_module_schema", node=MNISTDataModuleSchema)
    cs.store(group="data_module", name="cifar10_data_module_schema", node=CIFAR10DataModuleSchema)
