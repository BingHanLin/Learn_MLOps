import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(cfg.model.name)
    print(cfg.training.max_epochs)


if __name__ == "__main__":
    main()