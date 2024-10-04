from datetime import datetime 
import yaml 
import re 


def get_job_name_with_datetime(prefix: str) -> str:
    """
    Gets a job name by adding current time to prefix.
    
    Args:
        prefix: A string of job name prefix.
    Returns:
        A job name with added date time stamp.
    """
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{prefix}-{now}"
    return job_name


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None