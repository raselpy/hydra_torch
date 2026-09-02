# hydra_torch

[![CI](https://github.com/raselpy/hydra_torch/actions/workflows/ci.yml/badge.svg)](https://github.com/raselpy/hydra_torch/actions/workflows/ci.yml)
[![CD](https://github.com/raselpy/hydra_torch/actions/workflows/cd.yml/badge.svg)](https://github.com/raselpy/hydra_torch/actions/workflows/cd.yml)

**A modular, reproducible deep learning training framework** built with Hydra, PyTorch Lightning, MLflow and DVC.

Designed for clean experiment management, structured configuration, and production-oriented workflows.

---

## Key Features

- **Hydra + structured configs** — Fully typed ConfigStore schemas, composable YAML groups, and `instantiate()` for all components
- **PyTorch Lightning** — Clean training loops, DataModules, callbacks, and checkpointing
- **Modular model design** — Backbone → Adapter → Head architecture (easy to swap ResNet, custom heads, etc.)
- **Experiment tracking + Model Registry** — MLflow integration with resolved-config and data-version (`dvc.lock`) logging, plus **accuracy-gated champion promotion**: a new model version is only promoted to the `champion` alias if it beats the current champion's test accuracy
- **Reproducible pipelines** — DVC stages for `prepare_data → train → evaluate`, with the exact data version logged to MLflow on every run
- **Docker support** — GPU-ready training/serving image (PyTorch 2.8 + CUDA 12.9) orchestrated via docker-compose with a real MLflow tracking server, plus a separate lightweight image for CI tests
- **CI/CD** — GitHub Actions runs lint + tests + coverage enforcement on every push; on success, a second workflow builds and publishes the Docker image to GHCR
- **80%+ test coverage** — enforced via `pytest-cov`; CI fails if coverage regresses below the current threshold
- **Out-of-the-box datasets** — MNIST and CIFAR-10 DataModules
- **Model serving** — FastAPI inference endpoint that loads the current `champion` model directly from the MLflow Model Registry

---

## Tech Stack

| Component              | Technology                          |
| ----------------------- | ------------------------------------ |
| Configuration           | Hydra + OmegaConf + Pydantic         |
| Training framework      | PyTorch Lightning                    |
| Deep learning            | PyTorch + Torchvision                |
| Experiment tracking      | MLflow (tracking + Model Registry)   |
| Pipeline / versioning    | DVC                                  |
| Serving                  | FastAPI + Uvicorn                    |
| Containerization         | Docker (CUDA 12.9) + docker-compose  |
| Testing / CI             | pytest + pytest-cov + ruff + GitHub Actions |
| CD / image registry      | GitHub Actions + GitHub Container Registry (GHCR) |

---

## Quickstart

### 1. Install

```bash
pip install -e ".[dev,serve]"
```

> Both extras are required locally: `dev` for pytest/ruff/pre-commit, `serve` for the FastAPI serving dependencies exercised by `tests/test_serve.py`.

### 2. Run a single training job

```bash
python main.py                           # uses config.yaml defaults (CIFAR10, GPU)
python main.py trainer=cpu               # override any config group from the CLI
python main.py data_module=mnist task=mnist_classification
```

Outputs (checkpoints, logs) land in `outputs/<date>/<time>/`.

### 3. Run the full reproducible pipeline (DVC)

```bash
dvc repro
```

Runs `prepare_data → train → evaluate` in order, using `params.yaml` for hyperparameters (not `config.yaml`'s defaults — see note below). Only re-runs stages whose dependencies actually changed.

```bash
dvc dag             # view the pipeline graph
dvc metrics show    # view tracked metrics from train/evaluate
```

> **Note:** `params.yaml` drives `dvc repro`; `configs/config.yaml`'s defaults drive manual `python main.py` runs. These can silently diverge if you change one without the other.

### 4. Run with Docker + MLflow (GPU)

```bash
docker compose up --build
```

Brings up an MLflow tracking server and a GPU training container together. Once running:

- MLflow UI: <http://localhost:5000>
- Training logs stream live in the same terminal

To register a trained checkpoint into the Model Registry manually:

```bash
docker compose run --rm register
```

> Requires `nvidia-container-toolkit` (WSL2 + Docker Desktop GPU support on Windows).

### 5. Pull the pre-built image

Every push to `master` that passes CI automatically builds and publishes an image to GHCR:

```bash
docker pull ghcr.io/raselpy/hydra_torch:latest
```

Tags are also published per-commit (`ghcr.io/raselpy/hydra_torch:<commit-sha>`) for fully reproducible pulls.

### 6. Run the tests

```bash
pytest -v --cov=src --cov-report=term-missing
ruff check .
```

---

## Reproducibility

Every training run logs everything needed to reproduce it later, directly to MLflow:

- **Code** — MLflow auto-tags the run with the git commit SHA
- **Config** — the fully-resolved Hydra config is logged as `resolved_config.yaml`
- **Data** — `dvc.lock` (the exact DVC-tracked data hash) is logged as an artifact
- **Model** — trained weights are logged and registered under the Model Registry

To reproduce a specific run: open it in the MLflow UI, note the git commit and download `resolved_config.yaml` / `dvc.lock`, then:

```bash
git checkout <commit-sha>
dvc checkout
python main.py <overrides from resolved_config.yaml>
```

### Champion promotion

After each training run, the newly trained model is registered as a new version. It is only promoted to the `champion` alias (the version `serve.py` actually loads) if its test accuracy beats the current champion's. Otherwise the version is kept in the registry for history, but `champion` stays put.

---

## Project Structure

```
hydra_torch/
├── configs/                       # Hydra config groups
│   ├── data_module/
│   ├── task/
│   ├── trainer/
│   ├── logger/
│   └── config.yaml
├── src/
│   ├── config_schema/              # Typed ConfigStore schemas (mirrors configs/)
│   └── hydra_torch/
│       ├── data/
│       │   ├── data_modules.py     # MNIST / CIFAR10 LightningDataModules
│       │   └── transforms.py
│       ├── models/
│       │   ├── model.py            # Backbone + Adapter + Head composition
│       │   ├── backbones.py        # ResNet18 / ResNet50
│       │   ├── adapters.py
│       │   └── heads.py
│       ├── tasks.py                 # LightningModules (training/val/test steps)
│       ├── serving/
│       │   ├── serve.py            # FastAPI inference endpoint
│       │   └── register_model.py   # Manual checkpoint -> registry registration
│       └── scripts/
│           ├── train.py            # Training entrypoint + champion promotion
│           ├── prepare_data.py
│           └── evaluate.py
├── tests/                          # pytest suite (80%+ coverage, enforced in CI)
├── .github/workflows/
│   ├── ci.yml                      # lint + test + coverage on every push/PR
│   └── cd.yml                      # builds & publishes Docker image to GHCR on CI success
├── main.py                         # Entry point (thin wrapper -> scripts/train.py)
├── dvc.yaml                        # Pipeline definition
├── params.yaml                     # DVC parameters
├── Dockerfile                      # GPU training/serving image
├── Dockerfile.test                 # Lightweight image for running tests
├── docker-compose.yaml             # MLflow server + GPU training + register services
└── pyproject.toml
```

---

## Known Limitations

- **MNIST + ResNet18 channel mismatch:** `SimpleModel`'s ResNet18 backbone expects 3-channel input, but real MNIST images are 1-channel. This is tracked as an active, documented test failure (`tests/test_models.py`, `xfail(strict=True)`) rather than a silent gap — `task=mnist_classification` will fail at the model's first forward pass until this is resolved.
- **DVC remote is a local filesystem path** (`.dvc/config`), currently machine-specific. If you clone this repo, update `.dvc/config`'s remote URL to a path that exists on your machine before running `dvc pull`/`dvc repro`.
- **Coverage is at 80%, not 100%:** `train.py` (the main training loop) is the largest remaining gap — it's integration-heavy and not fully covered by fast unit tests. `serve.py` and `evaluate.py` also have partial coverage.