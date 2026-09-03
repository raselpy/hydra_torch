# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- CI/CD pipeline via GitHub Actions: `ci.yml` runs lint (`ruff`), tests, and coverage on every push/PR; `cd.yml` builds and publishes a Docker image to GitHub Container Registry (GHCR) on CI success
- Accuracy-gated champion promotion in `train.py` — a newly trained model is only promoted to the `champion` MLflow Model Registry alias if its test accuracy beats the current champion's; otherwise it's kept as a versioned entry only
- `dvc.lock` is now logged as an MLflow artifact on every training run, so the exact data version is reproducible alongside the resolved config and git commit
- Test coverage enforcement via `pytest-cov`, with `fail_under` in `pyproject.toml` raised incrementally as new tests were added: 62% → 69% → 77% → 80%
- New unit tests covering `tasks.py`, `register_model.py`, `transforms.py`, `transform_schema.py`, and additional `data_modules.py` coverage (`val_dataloader`, `prepare_data`)
- `/health` endpoint in `serve.py` now returns HTTP 503 when the model hasn't loaded yet (previously always returned 200), following standard liveness/readiness probe conventions; covered by new tests
- Dependabot configuration (`.github/dependabot.yml`) for weekly `pip`, `github-actions`, and `docker` dependency update checks
- README overhaul: architecture diagrams (rendered as images), Quickstart, Configuration, Experiment Tracking, Reproducibility, Testing, Known Limitations, and Future Improvements sections
- Standard Git PR workflow adopted going forward: feature branch → PR → CI check → merge, rather than direct pushes to `master`

### Fixed
- `SimpleModelschema` was defined but never registered with Hydra's `ConfigStore`, causing `task=mnist_classification` to fail at config-compose time
- `python-multipart` was missing from declared dependencies, causing `test_serve.py` to fail at collection time in CI despite passing locally (it happened to already be installed in the local conda environment)
- `train.py`'s champion-promotion logic raised `UnboundLocalError` on the very first run (no existing champion to compare against) because `new_acc` was only assigned inside a `try` block that raised before reaching it
- `pytest-cov` was missing from the `dev` extra in `pyproject.toml`, causing `--cov` flags to fail with "unrecognized arguments" in CI while working locally (only because it had been installed manually, outside `pyproject.toml`)
- `httpx2` was missing from the `dev` extra, causing `starlette.testclient.TestClient`-based tests to fail in CI with a clear `RuntimeError` while passing locally
- `Dockerfile` did not copy `dvc.lock` into the training image, so the DVC-lock-artifact-logging feature silently no-opped inside Docker even though it worked when run directly
- A soft-deleted MLflow experiment (`hydra_torch_runs`) blocked the SQLite backend from re-creating an experiment of the same name, until it was explicitly restored (`mlflow experiments restore`)
- Corrected a typo (`CIFAR10ModelSchemq` → `CIFAR10ModelSchema`) in the model schema class name, its `ConfigStore` registration, and the corresponding YAML default reference

### Changed
- CI's dependency install step now installs both the `dev` and `serve` extras (`pip install -e ".[dev,serve]"`), previously only `dev`, which had been silently excluding `test_serve.py`'s FastAPI dependencies

---

## How to use this file going forward

Add new entries under `[Unreleased]` as work happens. When cutting an actual release/tag, rename `[Unreleased]` to a version and date (e.g. `[0.2.0] - 2026-09-15`), and start a fresh `[Unreleased]` section above it.
