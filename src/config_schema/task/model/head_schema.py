from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class HeadConfig:
    _target_: str = MISSING


@dataclass
class IdentityHeadSchema(HeadConfig):
    _target_: str = "hydra_torch.models.heads.IdentityHead"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/head", name="identity_head_schema", node=IdentityHeadSchema)
