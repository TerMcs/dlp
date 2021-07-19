"""
Based on this template: https://github.com/ashleve/lightning-hydra-template
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from dlp.train import pipeline

@hydra.main(config_path="../../config/", config_name="configs.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Train model
    return pipeline.train_model(cfg)


if __name__ == "__main__":
    main()