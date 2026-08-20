# main.py
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl

# Registers all schemas into Hydra's ConfigStore
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

    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        val_acc = (
            trainer.callback_metrics.get("validation_accuracy")
            or trainer.callback_metrics.get("val_accuracy")
        )
        if train_loss is not None and val_acc is not None:
            log.info(f"Epoch {trainer.current_epoch} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    data_module = instantiate(cfg.data_module)
    task = instantiate(cfg.task)

    # Optional MLflow logger
    logger = instantiate(cfg.logger) if "logger" in cfg else None

    # Hydra-style instantiation of the Trainer (handles _target_ correctly)
    trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=[EpochLogger(cfg=cfg)],
    )

    log.info("Starting training loop...")
    trainer.fit(task, datamodule=data_module)

    log.info("Starting testing loop...")
    trainer.test(task, datamodule=data_module)


if __name__ == "__main__":
    main()