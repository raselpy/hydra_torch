from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class MLFlowLoggerSchema(LoggerConfig):
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "hydra_torch_runs"
    tracking_uri: str = "./mlruns"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="logger", name="mlflow_logger_schema", node=MLFlowLoggerSchema)
