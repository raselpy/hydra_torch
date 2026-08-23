import json
import logging
import os
import shutil

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# Load-bearing side-effecting import — see PLANNER.md section 1a.
# Do not remove: this registers cpu/gpu_trainer_schema, loss_function_schema,
# mlflow_logger_schema, etc. into Hydra's ConfigStore.
from src.config_schema import setup_config  # noqa: F401

log = logging.getLogger(__name__)

# Absolute path anchored to this file's own location. A relative config_path
# resolves against the __main__ script, NOT this file, when this
# hydra.main-decorated function is imported and called from elsewhere
# (which is exactly what main.py does). See PLANNER.md section 1d.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "configs")


class EpochLogger(Callback):
    """Permanent one-line-per-epoch console log (progress bar overwrites
    itself in place and doesn't survive in scrollback)."""

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        log.info(
            f"Epoch {trainer.current_epoch}: "
            f"train_loss={metrics.get('train_loss_epoch', 'n/a')}, "
            f"train_acc={metrics.get('train_accuracy', 'n/a')}, "
            f"val_acc={metrics.get('validation_accuracy', 'n/a')}"
        )


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Reproducibility
    if "seed" in cfg:
        pl.seed_everything(cfg.seed, workers=True)

    # 2. Instantiate Lightning components + the MLflow logger
    data_module = instantiate(cfg.data_module)
    task = instantiate(cfg.task)
    mlflow_logger = instantiate(cfg.logger)

    # Start every run with a clean checkpoints/ dir. Without this,
    # ModelCheckpoint(filename="best") never overwrites an existing file —
    # Lightning auto-versions instead (best.ckpt, best-v1.ckpt, best-v2.ckpt,
    # ...), which silently breaks dvc.yaml's evaluate stage: it always
    # points at the literal "checkpoints/best.ckpt", so it would keep
    # testing the FIRST-ever run's checkpoint forever, not the latest one.
    #
    # NOTE: clear only the CONTENTS of checkpoints/, not the directory
    # itself. In docker-compose, checkpoints/ is a bind mount — Linux
    # refuses to rmdir an active mount point (OSError: Device or resource
    # busy), even from inside the container. Deleting files/subdirs inside
    # it is fine; removing the mount-point directory itself is not.
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    for entry in os.listdir(checkpoint_dir):
        entry_path = os.path.join(checkpoint_dir, entry)
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best",
        monitor="validation_accuracy",
        mode="max",
        save_last=True,
    )

    trainer = instantiate(
        cfg.trainer,
        callbacks=[EpochLogger(), checkpoint_callback],
        logger=mlflow_logger,
    )

    # 3. Log the fully-resolved config as an MLflow artifact, plus a couple
    # of top-level tags, so every run is reproducible from its own tracking
    # record and filterable in the UI.
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow_logger.experiment.log_text(mlflow_logger.run_id, resolved_cfg, "resolved_config.yaml")
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "experiment_name", cfg.experiment_name)
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "seed", cfg.seed)

    # 4. Train
    log.info("Starting training loop...")
    trainer.fit(task, datamodule=data_module)

    # 5. Test the BEST checkpoint (by validation_accuracy), not just the
    # final-epoch in-memory weights.
    log.info("Starting testing loop...")
    test_results = trainer.test(task, datamodule=data_module, ckpt_path="best")

    # 6. Write metrics.json for DVC's `metrics:` tracking in the train stage.
    with open("metrics.json", "w") as f:
        json.dump(test_results[0], f, indent=2)


if __name__ == "__main__":
    main()
