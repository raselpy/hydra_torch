from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

from src.config_schemas import data_module_schema,model_module_schema


@dataclass
class Config:
    data:  data_module_schema.DataModuleConfig = MISSING
    model: model_module_schema.ModelModuleConfig = MISSING


def setup_config() -> None:
    data_module_schema.setup_config()
    model_module_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)