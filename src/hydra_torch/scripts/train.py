import json
import logging
import os
import shutil

import hydra
import mlflow
import mlflow.pytorch
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


def clear_checkpoint_dir(checkpoint_dir: str) -> None:
    """Start every run with a clean checkpoints/ dir. Without this,
    ModelCheckpoint(filename="best") never overwrites an existing file —
    Lightning auto-versions instead (best.ckpt, best-v1.ckpt, best-v2.ckpt,
    ...), which silently breaks dvc.yaml's evaluate stage: it always
    points at the literal "checkpoints/best.ckpt", so it would keep
    testing the FIRST-ever run's checkpoint forever, not the latest one.

    NOTE: clear only the CONTENTS of checkpoint_dir, not the directory
    itself. In docker-compose, checkpoints/ is a bind mount — Linux
    refuses to rmdir an active mount point (OSError: Device or resource
    busy), even from inside the container. Deleting files/subdirs inside
    it is fine; removing the mount-point directory itself is not.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    for entry in os.listdir(checkpoint_dir):
        entry_path = os.path.join(checkpoint_dir, entry)
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)


def log_run_artifacts(mlflow_logger, cfg: DictConfig, dvc_lock_path: str) -> None:
    """Log the fully-resolved config as an MLflow artifact, plus a couple
    of top-level tags, so every run is reproducible from its own tracking
    record and filterable in the UI. Also logs dvc.lock so this run's
    exact data version is reproducible — DVC records the output hash for
    data/${data_module} here after `dvc repro`. To reproduce:
    `git checkout <commit>` + `dvc checkout` using this exact dvc.lock.
    """
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow_logger.experiment.log_text(mlflow_logger.run_id, resolved_cfg, "resolved_config.yaml")
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "experiment_name", cfg.experiment_name)
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "seed", cfg.seed)

    if os.path.exists(dvc_lock_path):
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, dvc_lock_path)
    else:
        log.warning("dvc.lock not found — data version will not be logged for this run.")


def register_and_promote_champion(cfg: DictConfig, mlflow_logger, task, test_results) -> None:
    """Register the trained model to MLflow's Model Registry, so serve.py
    can load it by a stable alias instead of a hardcoded checkpoint path.

    Model Registry REQUIRES a database-backed tracking store (SQLite/
    Postgres/MySQL) — it does NOT work against a plain file:// store.
    Bare-metal runs (MLFLOW_TRACKING_URI unset) default to file:./mlruns
    and skip this step entirely rather than crash; only the compose
    SQLite-backed mlflow-server (MLFLOW_TRACKING_URI=http://mlflow-server:5000)
    actually supports it. See PLANNER.md section 1j.

    Design simplification, stated explicitly rather than hidden: a new
    version is only promoted to the "champion" alias if its test accuracy
    beats the current champion's — a real production setup might gate
    promotion behind additional validation checks or human review, not
    accuracy alone. Out of scope here.
    """
    tracking_uri = cfg.logger.tracking_uri
    if tracking_uri.startswith("file:"):
        log.warning(
            f"Skipping model registration: tracking_uri={tracking_uri} is a "
            f"file-based store, which does not support MLflow Model Registry. "
            f"Run via `docker compose up train` (SQLite-backed mlflow-server) "
            f"to register a model."
        )
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)

        backbone_name = type(task.model.backbone).__name__
        adapter_name = type(task.model.adapter).__name__

        registered_model_name = f"hydra_torch_{backbone_name}"

        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_param("backbone", backbone_name)
            mlflow.log_param("adapter", adapter_name)

            model_info = mlflow.pytorch.log_model(
                task.model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )

        client = mlflow.tracking.MlflowClient()

        new_acc = test_results[0].get("test_accuracy")
        should_promote = True
        champion_acc = None
        try:
            current_champion = client.get_model_version_by_alias(registered_model_name, "champion")
            champion_run = client.get_run(current_champion.run_id)
            champion_acc = champion_run.data.metrics.get("test_accuracy")
            if champion_acc is not None and new_acc is not None:
                should_promote = new_acc > champion_acc
        except mlflow.exceptions.RestException:
            should_promote = True

        if should_promote:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="champion",
                version=model_info.registered_model_version,
            )
            log.info(f"New champion: version {model_info.registered_model_version} (acc={new_acc})")
        else:
            log.info(f"Run not promoted: acc={new_acc} <= current champion acc={champion_acc}")

    except Exception:
        log.warning(
            "Model registration failed; training results are unaffected.",
            exc_info=True,
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

    clear_checkpoint_dir("checkpoints")

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

    dvc_lock_path = os.path.join(_THIS_DIR, "..", "..", "..", "dvc.lock")
    log_run_artifacts(mlflow_logger, cfg, dvc_lock_path)

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

    # 7. Register + accuracy-gated champion promotion.
    register_and_promote_champion(cfg, mlflow_logger, task, test_results)


if __name__ == "__main__":
    main()