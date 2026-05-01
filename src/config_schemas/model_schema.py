from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

@dataclass
class ModelConfig:
    _target_: str = "src.modular.model.Net"

def setup_config():
    cs = ConfigStore.instance()
    cs.store(group="model", name="model_schema", node=ModelConfig)