import hydra
from omegaconf import OmegaConf, DictConfig
from src.config_schemas.config_schema import setup_config
setup_config()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
