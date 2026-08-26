from hydra.core.config_store import ConfigStore

from .adapter_schema import AdapterConfig
from .backbone_schema import BackboneConfig
from .data_module_schema import CIFAR10DataModuleSchema, MNISTDataModuleSchema
from .head_schema import HeadConfig
from .logger_schema import MLFlowLoggerSchema
from .loss_function_schema import LossFunctionConfig
from .model_schema import ModelConfig
from .optimizer_schema import OptimizerConfig
from .task_schema import CIFAR10TaskSchema, MNISTTaskSchema
from .trainer_schema import CPUTrainerSchema, GPUTrainerSchema
from .transform_schema import MNISTTransformSchema as MNISTTransformSchema


def setup_config() -> None:
    cs = ConfigStore.instance()
    # logger
    cs.store(group="logger", name="mlflow_logger_schema", node=MLFlowLoggerSchema)
    # data_module
    cs.store(group="data_module", name="mnist_data_module_schema", node=MNISTDataModuleSchema)
    cs.store(group="data_module", name="cifar10_data_module_schema", node=CIFAR10DataModuleSchema)
    # trainer
    cs.store(group="trainer", name="cpu_trainer_schema", node=CPUTrainerSchema)
    cs.store(group="trainer", name="gpu_trainer_schema", node=GPUTrainerSchema)
    # task
    cs.store(group="task", name="mnist_task_schema", node=MNISTTaskSchema)
    cs.store(group="task", name="cifar10_task_schema", node=CIFAR10TaskSchema)
    # nested under task/
    cs.store(group="task/model", name="model_schema", node=ModelConfig)
    cs.store(group="task/optimizer", name="optimizer_schema", node=OptimizerConfig)
    cs.store(group="task/loss_function", name="loss_function_schema", node=LossFunctionConfig)
    # nested under task/model/ (Phase: backbone/adapter/head split into separate groups)
    cs.store(group="task/model/backbone", name="backbone_schema", node=BackboneConfig)
    cs.store(group="task/model/adapter", name="adapter_schema", node=AdapterConfig)
    cs.store(group="task/model/head", name="head_schema", node=HeadConfig)


# Auto-register when the package is imported
setup_config()
