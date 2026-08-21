import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config_schema import setup_config  # noqa: F401

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "..", "configs")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    data_module = instantiate(cfg.data_module)
    data_module.prepare_data()
    print(f"Data prepared at: {cfg.data_module.data_dir}")


if __name__ == "__main__":
    main()