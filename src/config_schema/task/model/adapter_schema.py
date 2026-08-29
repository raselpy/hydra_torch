from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class AdapterConfig:
    _target_: str = MISSING
    in_features: int = MISSING
    out_features: int = MISSING


@dataclass
class LinearAdapterSchema(AdapterConfig):
    _target_: str = "hydra_torch.models.adapters.LinearAdapter"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/adapter", name="linear_adapter_schema", node=LinearAdapterSchema)
