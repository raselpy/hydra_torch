"""Register a trained checkpoint into the MLflow Model Registry and assign
it an alias (default: "champion") that serve.py loads from.

This is a manual step you run *after* training produces a checkpoint — it
does not run automatically as part of `main.py` / `scripts/train.py`.

Usage (host machine, with `docker compose up -d mlflow-server` running):
    MLFLOW_TRACKING_URI=http://localhost:5000 \
        python -m src.hydra_torch.register_model --checkpoint-path checkpoints/best.ckpt

Usage (inside the `register` compose service — MLFLOW_TRACKING_URI is set
there to http://mlflow-server:5000 automatically):
    docker compose run --rm register
"""

from __future__ import annotations

import argparse
import logging
import os

import mlflow
import mlflow.pytorch
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from mlflow import MlflowClient

# Load-bearing side-effecting import — registers Hydra ConfigStore schemas.
from src.config_schema import setup_config  # noqa: F401

log = logging.getLogger(__name__)

# This file lives at src/hydra_torch/register_model.py — same depth as
# serve.py, two levels under the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "configs")

# Must match serve.py's REGISTERED_MODEL_NAME / MODEL_ALIAS env var defaults
# exactly — these two files coordinate through the same registry entries.
DEFAULT_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "hydra_torch_CIFAR10Model")
DEFAULT_ALIAS = os.environ.get("MODEL_ALIAS", "champion")


def _build_model_from_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train a model first "
            f"(see README Quickstart) or pass --checkpoint-path explicitly."
        )

    with initialize_config_dir(config_dir=_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["task=cifar10_classification", "data_module=cifar10"],
        )
        model = instantiate(cfg.task.model)

    # weights_only=False: PyTorch Lightning checkpoints carry non-tensor
    # state (optimizer states, callback states, hyperparameters), and
    # PyTorch >= 2.6 defaults torch.load to weights_only=True, which cannot
    # unpickle those. Safe here since this loads our own checkpoint, not an
    # untrusted third-party file. Same rationale as serve.py's prior direct
    # checkpoint loading (Phase 10).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    return model


def register(checkpoint_path: str, model_name: str, alias: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Model Registry access requires "
            "a database-backed MLflow tracking server — run via docker-compose "
            "(sets this automatically) or export it yourself, e.g. "
            "MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"Using MLflow tracking URI: {tracking_uri}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model_from_checkpoint(checkpoint_path, device)

    with mlflow.start_run(run_name="register_from_checkpoint") as run:
        mlflow.log_param("source_checkpoint", checkpoint_path)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    log.info(f"Logged model under run {run_id} and registered it as '{model_name}'.")

    # Find the version that was just created by this run (avoids depending
    # on any particular return-value shape from log_model across mlflow
    # versions — we just ask the registry which version has this run_id).
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
        raise RuntimeError(
            f"Registered model '{model_name}' but could not find a version "
            f"linked to run {run_id} — registry state may be inconsistent."
        )
    version = max(int(v.version) for v in matching)

    # set_registered_model_alias moves the alias automatically — an alias
    # can only point at one version at a time, so there's no separate
    # "archive the old one" step needed (unlike the legacy stage system).
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)

    # Verify, don't trust — read the alias back from the registry rather
    # than assuming the API call above actually stuck (this project has
    # been bitten before by treating a call's return/non-error as proof of
    # effect — see PLANNER.md Phase 10 lessons).
    resolved = client.get_model_version_by_alias(model_name, alias)
    if resolved.version != str(version):
        raise RuntimeError(
            f"Requested alias '{alias}' -> version {version} for '{model_name}', "
            f"but the registry now resolves '@{alias}' to version "
            f"{resolved.version} instead. The alias assignment did not take "
            f"effect as expected."
        )
    log.info(f"Confirmed: '{model_name}@{alias}' now resolves to version {resolved.version}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("CHECKPOINT_PATH", "checkpoints/best.ckpt"),
        help="Path to the Lightning checkpoint to register (default: %(default)s)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered model name in the MLflow Model Registry (default: %(default)s)",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help="Alias to point at the newly registered version (default: %(default)s)",
    )
    args = parser.parse_args()
    register(args.checkpoint_path, args.model_name, args.alias)


if __name__ == "__main__":
    main()
