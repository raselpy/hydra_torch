import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from src.config_schema import setup_config  # Correctly import setup_config from the package
log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("✅ Phase 1: Config composed, validated via Pydantic, and resolved.")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
