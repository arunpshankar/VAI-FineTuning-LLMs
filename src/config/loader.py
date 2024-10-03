from src.config.logging import logger 
from typing import Dict, Any, Optional
from glob import glob
import yaml
import os 

class Config:
    _instance = None
    _project_config = None  # Cache for the project configuration

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False  # Use single underscore
        return cls._instance
    
    def __init__(self, model_name: Optional[str] = None, config_dir: str = "./configs", reinitialize: bool = False):
        """
        Initialize the Config class.

        Args:
            model_name (Optional[str]): Name of the model to load configurations for. If None, only project.yml is loaded.
            config_dir (str): Path to the directory containing YAML configuration files.
            reinitialize (bool): If True, forces reinitialization of the configuration. Default is False.
        """
        if self._initialized and not reinitialize:
            return

        self._initialized = True
        self.config_dir = config_dir
        self.model_name = model_name

        # Load configurations
        self.__config = self._load_configs()

        # Set attributes dynamically based on config keys
        for key, value in self.__config.items():
            setattr(self, key.upper(), value)

        # Optionally set Google application credentials if provided
        credentials_path = getattr(self, 'CREDENTIALS_PATH', None)
        if credentials_path:
            self._set_google_credentials(credentials_path)

    def _load_configs(self) -> Dict[str, Any]:
        """
        Load project-level and model-specific configurations if applicable.

        Returns:
            dict: Merged configuration data.
        """
        config = self._load_project_config()
        
        if self.model_name:
            model_config = self._load_model_config()
            config = self._merge_dicts(config, model_config)
        
        return config

    def _load_project_config(self) -> Dict[str, Any]:
        """
        Load the project-level configuration from 'project.yml'.

        Returns:
            dict: Project-level configuration data.
        """
        if Config._project_config is not None:
            logger.info("Using cached project configuration.")
            return Config._project_config

        project_config_path = os.path.join(self.config_dir, 'project.yml')
        try:
            with open(project_config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                logger.info(f"Loaded project configuration from {project_config_path}")
                Config._project_config = config_data if config_data else {}
                return Config._project_config
        except FileNotFoundError:
            logger.error(f"Project configuration file not found at {project_config_path}")
            Config._project_config = {}
            return Config._project_config
        except Exception as e:
            logger.error(f"Failed to load project configuration. Error: {e}")
            raise

    def _load_model_config(self) -> Dict[str, Any]:
        """
        Load model-specific YAML configuration files.

        Returns:
            dict: Model-specific configuration data.
        """
        model_config = {}
        model_config_dir = os.path.join(self.config_dir, self.model_name)
        if not os.path.isdir(model_config_dir):
            logger.error(f"Model configuration directory does not exist: {model_config_dir}")
            return model_config

        try:
            # Find all YAML files in the model's configuration directory
            yaml_files = glob(os.path.join(model_config_dir, '*.yaml')) + glob(os.path.join(model_config_dir, '*.yml'))
            if not yaml_files:
                logger.warning(f"No YAML configuration files found in directory: {model_config_dir}")

            for yaml_file in yaml_files:
                with open(yaml_file, 'r') as file:
                    data = yaml.safe_load(file)
                    if data:
                        model_config = self._merge_dicts(model_config, data)
                        logger.info(f"Loaded configuration from {yaml_file}")
            return model_config
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
        self.__config = self._load_configs()
        for key, value in self.__config.items():
            setattr(self, key.upper(), value)
        logger.info("Configuration refreshed.")

    def reset(self) -> None:
        """
        Reset all dynamic configuration attributes to None.
        """
        # Loop over all the attributes that were dynamically set by the configuration
        for key in self.__config.keys():
            setattr(self, key.upper(), None)
        logger.info("All configuration attributes have been reset to None.")

# Example usage:
# 1. Load configuration with a specific model name
# config_with_model = Config(model_name='custom_model')

# 2. Load only project configuration without mentioning a model name
# config_project_only = Config()  # No model_name provided

# 3. Load configuration with a specific model name
# config_default = Config(model_name='gemini_1_5')
