import logging
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Load-bearing side-effecting import — registers Hydra ConfigStore schemas.
from src.config_schema import setup_config  # noqa: F401

log = logging.getLogger(__name__)


class EpochLogger(pl.Callback):
    """Logs a permanent line per epoch and saves the resolved Hydra config as an MLflow artifact."""

    def __init__(self, cfg: DictConfig = None):
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module):
        if self.cfg and trainer.logger:
            for logger in trainer.loggers:
                if hasattr(logger, "experiment"):
                    logger.experiment.log_text(
                        logger.run_id,
                        OmegaConf.to_yaml(self.cfg),
                        "hydra_config.yaml",
                    )
                    logger.experiment.set_tag(logger.run_id, "experiment_name", self.cfg.experiment_name)
                    logger.experiment.set_tag(logger.run_id, "seed", self.cfg.seed)

    # NOTE: on_epoch_end was removed in PyTorch Lightning 2.0 — this must be
    # on_train_epoch_end, or the hook silently never fires.
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch") or metrics.get("train_loss")
        val_acc = metrics.get("validation_accuracy") or metrics.get("val_accuracy")
        log.info(
            f"Epoch {trainer.current_epoch} | "
            f"Loss: {train_loss if train_loss is None else float(train_loss):.4f} | "
            f"Val Acc: {val_acc if val_acc is None else float(val_acc):.4f}"
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.data_module)
    task = instantiate(cfg.task)
    logger = instantiate(cfg.logger) if "logger" in cfg else None

    # Fixed, predictable checkpoint path so DVC's train stage has a stable
    # outs: directory to track.
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best",
        monitor="validation_accuracy",
        mode="max",
        save_last=True,
    )

    trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[EpochLogger(cfg=cfg), checkpoint_callback],
    )

    log.info("Starting training loop...")
    trainer.fit(task, datamodule=data_module)

    log.info("Starting testing loop...")
    test_results = trainer.test(task, datamodule=data_module, ckpt_path="best")

    with open("metrics.json", "w") as f:
        json.dump(test_results[0], f, indent=2)


if __name__ == "__main__":
    main()