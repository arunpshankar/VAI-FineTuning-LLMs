import logging
import yaml
import os
from google.cloud import aiplatform

def setup_environment() -> None:
    """Sets up the Google Cloud project and initializes Vertex AI."""
    logging.info("Setting up the environment.")
    try:
        with open('configs/project_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        project_id = config['project_id']
        location = config['location']
        bucket_uri = f"gs://{config['bucket_name']}"

        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        os.environ['GOOGLE_CLOUD_REGION'] = location

        aiplatform.init(project=project_id, location=location, staging_bucket=bucket_uri)
        logging.info("Vertex AI initialized.")
    except Exception as e:
        logging.exception("Failed to set up the environment.")
        raise e
