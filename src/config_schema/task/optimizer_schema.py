from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    _partial_: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/optimizer", name="optimizer_schema", node=OptimizerConfig)
