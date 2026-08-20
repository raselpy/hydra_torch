from hydra.core.config_store import ConfigStore

from .logger_schema import MLFlowLoggerSchema
from .data_module_schema import MNISTDataModuleSchema, CIFAR10DataModuleSchema
from .trainer_schema import CPUTrainerSchema, GPUTrainerSchema
from .task_schema import MNISTTaskSchema, CIFAR10TaskSchema
from .model_schema import ModelConfig
from .optimizer_schema import OptimizerConfig
from .loss_function_schema import LossFunctionConfig
from .transform_schema import MNISTTransformSchema  


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


# Auto-register when the package is imported
setup_config()