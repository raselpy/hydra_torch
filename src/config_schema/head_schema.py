from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class HeadConfig:
    _target_: str = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/head", name="head_schema", node=HeadConfig)
