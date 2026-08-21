# hydra_torch

**A modular, reproducible deep learning training framework** built with Hydra, PyTorch Lightning, MLflow and DVC.

Designed for clean experiment management, structured configuration, and production-oriented workflows.

---

## Key Features

- **Hydra + structured configs** — Fully typed ConfigStore schemas, composable YAML groups, and `instantiate()` for all components
- **PyTorch Lightning** — Clean training loops, DataModules, callbacks, and checkpointing
- **Modular model design** — Backbone → Adapter → Head architecture (easy to swap ResNet, custom heads, etc.)
- **Experiment tracking** — MLflow integration with resolved config logging and metric tracking
- **Reproducible pipelines** — DVC stages for `prepare_data → train → evaluate`
- **Docker support** — GPU-ready image based on PyTorch 2.8 + CUDA 12.9
- **Out-of-the-box datasets** — MNIST and CIFAR-10 DataModules

---

## Tech Stack

| Component              | Technology                          |
|------------------------|-------------------------------------|
| Configuration          | Hydra + OmegaConf + Pydantic        |
| Training framework     | PyTorch Lightning                   |
| Deep learning          | PyTorch + Torchvision               |
| Experiment tracking    | MLflow                              |
| Pipeline / versioning  | DVC                                 |
| Containerization       | Docker (CUDA 12.9)                  |

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
│   ├── config_schema/        # Typed ConfigStore schemas
│   └── hydra_torch/
│       ├── data_modules.py
│       ├── models.py         # Backbone + Adapter + Head
│       ├── backbones.py      # ResNet18 / ResNet50
│       ├── adapters.py
│       ├── heads.py
│       ├── tasks.py          # LightningModules
│       ├── transforms.py
│       └── scripts/
│           ├── train.py
│           ├── prepare_data.py
│           └── evaluate.py
├── main.py                   # Entry point
├── dvc.yaml                  # Pipeline definition
├── params.yaml               # DVC parameters
├── Dockerfile
└── pyproject.toml