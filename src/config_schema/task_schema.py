from typing import Any, Dict, Optional
from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING
from config_schema import model_schema, optimizer_schema, loss_function_schema


@dataclass
class TaskConfig:
    _target_: str = MISSING
    model: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    loss_function: Optional[Dict[str, Any]] = None


@dataclass
class MNISTTaskSchema(TaskConfig):
    _target_: str = "hydra_torch.tasks.MNISTClassificationTrainingTask"


@dataclass
class CIFAR10TaskSchema(TaskConfig):
    _target_: str = "hydra_torch.tasks.CIFAR10ClassificationTrainingTask"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="task", name="mnist_task_schema", node=MNISTTaskSchema)
    cs.store(group="task", name="cifar10_task_schema", node=CIFAR10TaskSchema)

    model_schema.setup_config()
    optimizer_schema.setup_config()
    loss_function_schema.setup_config()