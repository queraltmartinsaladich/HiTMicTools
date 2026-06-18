import yaml
import zipfile
from pathlib import Path
from typing import Dict, Any, Union, Optional

from HiTMicTools.utils import update_config


class ConfigLoader:
    """Enhanced configuration loader supporting YAML files and zip archives."""

    @staticmethod
    def load_config(
        config_path: Union[str, Path], override_args: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file or zip archive, with optional override.
        """
        config_path = Path(config_path)

        if config_path.suffix == ".zip":
            config = ConfigLoader._load_from_zip(config_path)
        elif config_path.suffix in [".yml", ".yaml"]:
            config = ConfigLoader._load_from_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Apply overrides if provided and not empty
        if override_args:
            config = update_config(config, override_args)
        return config

    @staticmethod
    def _load_from_yaml(yaml_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as config_file:
            return yaml.safe_load(config_file)

    @staticmethod
    def _load_from_zip(zip_path: Path) -> Dict[str, Any]:
        """Load configuration from zip archive."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Look for config file in zip
            config_files = [
                f for f in zip_ref.namelist() if f.endswith((".yml", ".yaml"))
            ]

            if not config_files:
                raise FileNotFoundError("No YAML config file found in zip archive")

            # Use first config file found
            with zip_ref.open(config_files[0]) as config_file:
                return yaml.safe_load(config_file)
