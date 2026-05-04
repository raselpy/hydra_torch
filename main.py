import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from src.config_schemas.config_schema import setup_config
setup_config()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.data)
    trainloader, testloader = dataset.getData()
    dataset.imgShow()
    model = instantiate(cfg.model)
    model = model.get_model()



if __name__ == "__main__":
    main()
