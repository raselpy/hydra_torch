# main.py
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# This import is NOT unused — importing src.config_schema runs its __init__.py,
# which calls setup_config() and registers cpu_trainer_schema, gpu_trainer_schema,
# and the data_module/task schemas into Hydra's ConfigStore. Without this import,
# Hydra cannot resolve `defaults: [gpu_trainer_schema]` (or cpu/task/data_module
# equivalents), and composition fails at import time, before @hydra.main even
# runs your function.
from src.config_schema import setup_config  # noqa: F401

log = logging.getLogger(__name__)


class EpochLogger(Callback):
    """Prints a permanent one-line summary per epoch to the console/log file,
    instead of relying on the progress bar (which overwrites itself in place
    and disappears from scrollback once the run finishes)."""

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        log.info(
            f"Epoch {trainer.current_epoch}: "
            f"train_loss={metrics.get('train_loss_epoch', 'n/a')}, "
            f"train_acc={metrics.get('train_accuracy', 'n/a')}, "
            f"val_acc={metrics.get('validation_accuracy', 'n/a')}"
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Reproducibility: set seed if provided in top-level config
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate Lightning components using Hydra's _target_ pattern
    data_module = instantiate(cfg.data_module)
    task = instantiate(cfg.task)
    trainer = instantiate(cfg.trainer, callbacks=[EpochLogger()])

    # 3. Run training
    log.info("Starting training loop...")
    trainer.fit(task, datamodule=data_module)

    # 4. Run evaluation on the held-out test set
    log.info("Starting testing loop...")
    trainer.test(task, datamodule=data_module)


if __name__ == "__main__":
    main()