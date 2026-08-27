from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from config_schema.task import loss_function_schema, optimizer_schema
from config_schema.task.model import model_schema


@dataclass
class TaskConfig:
    _target_: str = MISSING
    model: Optional[dict[str, Any]] = None
    optimizer: Optional[dict[str, Any]] = None
    loss_function: Optional[dict[str, Any]] = None


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
