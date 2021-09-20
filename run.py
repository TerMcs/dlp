"""
Based on this template: https://github.com/ashleve/lightning-hydra-template
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from dlp.train import dlppipeline

@hydra.main(config_path="config/", config_name="configs.yaml")
def main(cfg: DictConfig):

    # Train the model:
    return dlppipeline.train_model(cfg)


if __name__ == "__main__":
    main()