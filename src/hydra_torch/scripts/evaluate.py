import json
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config_schema import setup_config  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "configs")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    if "checkpoint_path" not in cfg:
        raise ValueError("Pass +checkpoint_path=<path>, e.g. +checkpoint_path=checkpoints/best.ckpt")

    data_module = instantiate(cfg.data_module)
    task = instantiate(cfg.task)
    trainer = instantiate(cfg.trainer, logger=False)

    results = trainer.test(task, datamodule=data_module, ckpt_path=cfg.checkpoint_path)

    with open("eval_metrics.json", "w") as f:
        json.dump(results[0], f, indent=2)
    print(f"Evaluation results written to eval_metrics.json: {results[0]}")


if __name__ == "__main__":
    main()
