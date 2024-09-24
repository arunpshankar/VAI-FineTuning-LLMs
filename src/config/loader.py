from src.config.logging import logger 
from typing import Dict
from typing import Any 
from glob import glob
import yaml
import os 


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, model_name: str, config_dir: str = "./configs"):
        """
        Initialize the Config class.

        Args:
            model_name (str): Name of the model to load configurations for (e.g., 'gemini').
            config_dir (str): Path to the directory containing YAML configuration files.
        """
        if self.__initialized:
            return
        self.__initialized = True

        self.config_dir = config_dir
        self.model_name = model_name

        # Load configurations
        self.__config = self._load_project_config()
        self.__config = self._load_model_configs(self.__config)

        # Set attributes dynamically based on config keys
        for key, value in self.__config.items():
            setattr(self, key.upper(), value)

        # Optionally set Google application credentials if provided
        credentials_path = getattr(self, 'CREDENTIALS_PATH', None)
        if credentials_path:
            self._set_google_credentials(credentials_path)

    def _load_project_config(self) -> Dict[str, Any]:
        """
        Load the project-level configuration from 'project.yml'.

        Returns:
            dict: Project-level configuration data.
        """
        project_config_path = os.path.join(self.config_dir, 'project.yml')
        try:
            with open(project_config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                logger.info(f"Loaded project configuration from {project_config_path}")
                return config_data if config_data else {}
        except FileNotFoundError:
            logger.error(f"Project configuration file not found at {project_config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load project configuration. Error: {e}")
            raise

    def _load_model_configs(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and merge model-specific YAML configuration files.

        Args:
            base_config (Dict[str, Any]): The base configuration to merge into.

        Returns:
            dict: Merged configuration data.
        """
        model_config_dir = os.path.join(self.config_dir, self.model_name)
        if not os.path.isdir(model_config_dir):
            logger.error(f"Model configuration directory does not exist: {model_config_dir}")
            return base_config

        try:
            # Find all YAML files in the model's configuration directory
            yaml_files = glob(os.path.join(model_config_dir, '*.yaml')) + glob(os.path.join(model_config_dir, '*.yml'))
            if not yaml_files:
                logger.warning(f"No YAML configuration files found in directory: {model_config_dir}")

            for yaml_file in yaml_files:
                with open(yaml_file, 'r') as file:
                    data = yaml.safe_load(file)
                    if data:
                        base_config = self._merge_dicts(base_config, data)
                        logger.info(f"Loaded configuration from {yaml_file}")
            return base_config
        except Exception as e:
            logger.error(f"Failed to load model-specific configuration files from {model_config_dir}. Error: {e}")
            raise

    @staticmethod
    def _merge_dicts(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Recursively merges two dictionaries.

        Args:
            a (Dict): The original dictionary.
            b (Dict): The dictionary to merge into a.

        Returns:
            Dict: The merged dictionary.
        """
        for key in b:
            if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = Config._merge_dicts(a[key], b[key])
            else:
                a[key] = b[key]
        return a

    @staticmethod
    def _set_google_credentials(credentials_path: str) -> None:
        """
        Set the Google application credentials environment variable.

        Args:
            credentials_path (str): Path to the Google credentials file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to {credentials_path}")

    def get(self, key: str, default=None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): Configuration key.
            default: Default value if key is not found.

        Returns:
            Any: Configuration value.
        """
        return getattr(self, key.upper(), default)

    def refresh(self) -> None:
        """
        Reload the configuration from files.

        Useful if configuration files have changed.
        """
        self.__config = self._load_project_config()
        self.__config = self._load_model_configs(self.__config)
        for key, value in self.__config.items():
            setattr(self, key.upper(), value)
        logger.info("Configuration refreshed.")


# Example usage:
# Initialize the Config singleton with the chosen model name
# For example, to load configurations for the 'gemini' model:

config = Config(model_name='gemini')