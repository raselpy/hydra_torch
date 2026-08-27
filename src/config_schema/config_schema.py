from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from config_schema import data_module_schema, logger_schema, trainer_schema
from config_schema.task import task_schema


@dataclass
class Config:
    experiment_name: str = "baseline"
    seed: int = 42
    data_module: data_module_schema.DataModuleConfig = MISSING
    task: task_schema.TaskConfig = MISSING
    trainer: trainer_schema.TrainerConfig = MISSING
    logger: logger_schema.LoggerConfig = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)

    data_module_schema.setup_config()
    trainer_schema.setup_config()
    task_schema.setup_config()
    logger_schema.setup_config()
