"""
Functionality for hydra configuration
"""

import os
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = "config"
CONFIG_FILE = "config.yaml"


def init_config(
    config_dir: str = CONFIG_DIR, config_file: str = CONFIG_FILE
) -> DictConfig:
    """
    Initialize hydra configuration

    Args:
    config_dir (str): Directory where the configuration file is located (relative to the project root)
    config_file (str): Name of the configuration file

    Returns:
    cfg (DictConfig): Hydra configuration object

    """

    abs_config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=abs_config_dir, version_base="1.1"):
        cfg = compose(config_name=CONFIG_FILE)

    return cfg


# monkey patch DictConfigs str function

DictConfig.__str__ = lambda self: OmegaConf.to_yaml(self, resolve=True)
