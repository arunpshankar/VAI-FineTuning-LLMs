from src.config.logging import logger
from src.config.loader import Config
from datetime import datetime
from typing import Optional
import yaml
import os


def get_job_name_with_datetime(prefix: str) -> str:
    """
    Generates a job name by appending the current timestamp to the given prefix.

    Args:
        prefix (str): The prefix to use for the job name.

    Returns:
        str: A job name with the current date and time appended.
    """
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{prefix}-{now}"
    logger.info(f"Generated job name: {job_name}")
    return job_name


def load_yaml(file_path: str) -> Optional[dict]:
    """
    Loads a YAML file and returns its contents.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Optional[dict]: The contents of the YAML file as a dictionary, or None if an error occurs.

    Raises:
        OSError: If the file cannot be opened.
        yaml.YAMLError: If the YAML content is invalid.
    """
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            logger.info(f"YAML file loaded successfully: {file_path}")
            return yaml_data
    except OSError as e:
        logger.error(f"Error opening YAML file: {file_path}. Error: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {file_path}. Error: {e}")
        raise


def setup_environment(config: Config) -> None:
    """
    Sets up environment variables for Google Cloud credentials using the provided configuration.

    Args:
        config (Config): The configuration object containing project details.

    Raises:
        KeyError: If the credentials_path is missing from the configuration.
    """
    credentials_path = config.PROJECT.get('credentials_path')
    if not credentials_path:
        logger.error("Credentials path not found in configuration")
        raise KeyError("Missing credentials_path in configuration")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    logger.info(f"Google Application Credentials set using {credentials_path}")


def load_hf_token(file_path: str) -> str:
    """
    Loads the Hugging Face token from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing the Hugging Face token.

    Returns:
        str: The Hugging Face token.

    Raises:
        KeyError: If the 'key' is missing in the YAML file.
        OSError, yaml.YAMLError: If there are issues loading or parsing the YAML file.
    """
    yaml_data = load_yaml(file_path)
    hf_token = yaml_data.get('key')
    if not hf_token:
        logger.error(f"Hugging Face token not found in YAML file: {file_path}")
        raise KeyError("Hugging Face token not found in YAML file")
    
    logger.info("Hugging Face token loaded successfully")
    return hf_token
