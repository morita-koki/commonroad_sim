from pathlib import Path
import omegaconf


# load yaml with omegaconf


class ConfigManager:
    workspace_dir: Path = None
    config_path: Path = None

    def __init__(self, workspace_dir: str, config_path: str):
        self.workspace_dir = Path(workspace_dir)
        self.config_path = Path(config_path)

    @staticmethod
    def load_config(config_path: str) -> omegaconf.DictConfig:
        return omegaconf.OmegaConf.load(config_path)
