from pydantic.dataclasses import dataclass


@dataclass
class LoggerConfig:
    _target_: str = "MISSING"


@dataclass
class MLFlowLoggerSchema(LoggerConfig):
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "hydra_torch_runs"
    tracking_uri: str = "./mlruns"
    # remove save_artifacts – it is not a valid argument
    # optional useful ones you can keep/add:
    # save_dir: str = "./mlruns"
    # log_model: bool = False
    # prefix: str = ""