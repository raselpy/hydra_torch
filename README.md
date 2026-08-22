# hydra_torch

![CI](https://github.com/raselpy/hydra_torch/actions/workflows/ci.yml/badge.svg)

**A modular, reproducible deep learning training framework** built with Hydra, PyTorch Lightning, MLflow and DVC.

Designed for clean experiment management, structured configuration, and production-oriented workflows.

---

## Key Features

- **Hydra + structured configs** — Fully typed ConfigStore schemas, composable YAML groups, and `instantiate()` for all components
- **PyTorch Lightning** — Clean training loops, DataModules, callbacks, and checkpointing
- **Modular model design** — Backbone → Adapter → Head architecture (easy to swap ResNet, custom heads, etc.)
- **Experiment tracking** — MLflow integration with resolved config logging and metric tracking
- **Reproducible pipelines** — DVC stages for `prepare_data → train → evaluate`
- **Docker support** — GPU-ready image based on PyTorch 2.8 + CUDA 12.9, orchestrated via docker-compose with a real MLflow tracking server
- **Tested & CI'd** — pytest suite + GitHub Actions running lint + tests on every push
- **Out-of-the-box datasets** — MNIST and CIFAR-10 DataModules

---

## Tech Stack

| Component              | Technology                          |
|------------------------|--------------------------------------|
| Configuration          | Hydra + OmegaConf + Pydantic        |
| Training framework     | PyTorch Lightning                   |
| Deep learning          | PyTorch + Torchvision               |
| Experiment tracking    | MLflow                              |
| Pipeline / versioning  | DVC                                 |
| Containerization       | Docker (CUDA 12.9) + docker-compose |
| Testing / CI           | pytest + GitHub Actions             |

---

## Quickstart

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run a single training job

```bash
python main.py                          # uses config.yaml defaults (CIFAR10, GPU)
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

- MLflow UI: [http://localhost:5000](http://localhost:5000)
- Training logs stream live in the same terminal

> Requires `nvidia-container-toolkit` (WSL2 + Docker Desktop GPU support on Windows).

### 5. Run the tests

```bash
pytest -v
ruff check .
```

---

## Project Structure

```text
hydra_torch/
├── configs/                  # Hydra config groups
│   ├── data_module/
│   ├── task/
│   ├── trainer/
│   ├── logger/
│   └── config.yaml
├── src/
│   ├── config_schema/         # Typed ConfigStore schemas
│   └── hydra_torch/
│       ├── data_modules.py
│       ├── models.py          # Backbone + Adapter + Head
│       ├── backbones.py       # ResNet18 / ResNet50
│       ├── adapters.py
│       ├── heads.py
│       ├── tasks.py           # LightningModules
│       ├── transforms.py
│       └── scripts/
│           ├── train.py
│           ├── prepare_data.py
│           └── evaluate.py
├── tests/                     # pytest suite
├── .github/workflows/ci.yml   # lint + test on every push/PR
├── main.py                    # Entry point (thin wrapper -> scripts/train.py)
├── dvc.yaml                   # Pipeline definition
├── params.yaml                # DVC parameters
├── Dockerfile
├── docker-compose.yml         # MLflow server + GPU training service
└── pyproject.toml
```

---

## Known Limitations

- **MNIST + ResNet18 channel mismatch:** `SimpleModel`'s ResNet18 backbone expects 3-channel input, but real MNIST images are 1-channel. This is tracked as an active, documented test failure (`tests/test_models.py`, `xfail(strict=True)`) rather than a silent gap — `task=mnist_classification` will fail at the model's first forward pass until this is resolved.
- **DVC remote is a local filesystem path** (`.dvc/config`), currently machine-specific. If you clone this repo, update `.dvc/config`'s remote URL to a path that exists on your machine before running `dvc pull`/`dvc repro`.