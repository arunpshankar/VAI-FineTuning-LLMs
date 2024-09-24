from src.config.logging import logger
from src.config.loader import config
from google.cloud import aiplatform
import vertexai


def setup_environment() -> None:
    """
    Sets up the Google Cloud project and initializes Vertex AI.
    """
    logger.info("Setting up the environment.")
    try:
        project_id = config.PROJECT.get('project_id')
        location = config.PROJECT.get('location')
        bucket_uri = f"gs://{config.PROJECT.get('bucket_name')}"

        aiplatform.init(project=project_id, location=location, staging_bucket=bucket_uri)
        vertexai.init(project=project_id, location=location)

        logger.info("Vertex AI initialized.")
    except Exception as e:
        logger.exception("Failed to set up the environment.")
        raise e
