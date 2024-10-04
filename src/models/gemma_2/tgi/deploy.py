from src.utils.common import get_job_name_with_datetime, load_yaml
from google.cloud.aiplatform import Endpoint, Model
from src.utils.quota import check_quota
from src.config.logging import logger
from src.config.loader import Config
from typing import Dict
import os


def set_environment_variables(config: Config, hf_token: str) -> Dict[str, str]:
    """
    Sets up the environment variables for the model container deployment.

    Args:
        config (Config): The configuration object containing model details.
        hf_token (str): Hugging Face token for authentication.

    Returns:
        Dict[str, str]: The environment variables to be used during deployment.
    """
    env_vars = {
        "MODEL_ID": config.MODEL.get('model_id'),
        "NUM_SHARD": config.DEPLOYMENT.get('accelerator_count'),
        "MAX_INPUT_LENGTH": config.MODEL.get('max_input_length'),
        "MAX_TOTAL_TOKENS": config.MODEL.get('max_total_tokens'),
        "MAX_BATCH_PREFILL_TOKENS": config.MODEL.get('max_batch_prefill_tokens'),
        "DEPLOY_SOURCE": "VERTEX_AI_SDK",
        "HF_TOKEN": hf_token
    }
    logger.info(f"Environment variables set: {env_vars}")
    return env_vars


def check_resource_quota(config: Config) -> None:
    """
    Checks the resource quota for the project.

    Args:
        config (Config): The configuration object containing project details.
    """
    logger.info("Checking resource quota")
    check_quota(
        project_id=config.PROJECT.get('project_id'),
        region=config.PROJECT.get('location'),
        accelerator_type=config.DEPLOYMENT.get('accelerator_type'),
        accelerator_count=config.DEPLOYMENT.get('accelerator_count'),
        is_for_training=False
    )


def create_endpoint(job_name: str, config: Config) -> Endpoint:
    """
    Creates a Vertex AI endpoint for the model deployment.

    Args:
        job_name (str): The job name to use for the endpoint.
        config (Config): The configuration object with deployment details.

    Returns:
        Endpoint: The created Vertex AI endpoint.
    """
    try:
        logger.info(f"Creating endpoint with job name: {job_name}")
        endpoint = Endpoint.create(
            display_name=f"{job_name}_endpoint",
            dedicated_endpoint_enabled=config.DEPLOYMENT.get('use_dedicated_endpoint')
        )
        logger.info(f"Endpoint {job_name}_endpoint created successfully")
        return endpoint
    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        raise


def upload_model(config: Config, job_name: str, env_vars: Dict[str, str]) -> Model:
    """
    Uploads the model to Vertex AI.

    Args:
        config (Config): The configuration object containing model and deployment details.
        job_name (str): The job name under which the model will be uploaded.
        env_vars (Dict[str, str]): Environment variables for the model container.

    Returns:
        Model: The uploaded Vertex AI model.

    Raises:
        Exception: If the model upload fails.
    """
    try:
        logger.info(f"Uploading model {job_name} with environment variables: {env_vars}")
        model = Model.upload(
            display_name=job_name,
            serving_container_image_uri=config.MODEL.get('tgi_docker_uri'),
            serving_container_ports=[8080],
            serving_container_environment_variables=env_vars,
            serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB shared memory
        )
        logger.info(f"Model {job_name} uploaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise


def deploy_model_to_endpoint(model: Model, endpoint: Endpoint, config: Config) -> None:
    """
    Deploys the uploaded model to the created Vertex AI endpoint.

    Args:
        model (Model): The model to be deployed.
        endpoint (Endpoint): The endpoint to which the model will be deployed.
        config (Config): The configuration object containing deployment details.

    Raises:
        Exception: If the model deployment fails.
    """
    try:
        logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
        model.deploy(
            endpoint=endpoint,
            machine_type=config.DEPLOYMENT.get('machine_type'),
            accelerator_type=config.DEPLOYMENT.get('accelerator_type'),
            accelerator_count=config.DEPLOYMENT.get('accelerator_count'),
            deploy_request_timeout=1800,  # 30 minutes timeout
        )
        logger.info(f"Model {model.display_name} deployed successfully to endpoint {endpoint.display_name}")
    except Exception as e:
        logger.error(f"Failed to deploy model to endpoint: {e}")
        raise


def run() -> None:
    """
    Main function to handle model upload and deployment to Vertex AI.
    
    It sets up the environment, checks resource quotas, loads the HF token,
    creates the endpoint, uploads the model, and deploys it to the created endpoint.
    """
    try:
        config = Config(model_name="gemma_2", reinitialize=True)
        setup_environment(config)
        
        model_name = config.MODEL.get('model_id')
        job_name = get_job_name_with_datetime(prefix=model_name)
        logger.info(f"Job name generated: {job_name}")

        # Check resource quotas
        check_resource_quota(config)

        # Create Vertex AI endpoint
        endpoint = create_endpoint(job_name, config)

        # Load HF token and set environment variables
        hf_token = load_hf_token('./credentials/hf.yml')
        env_vars = set_environment_variables(config, hf_token)

        # Upload the model to Vertex AI
        model = upload_model(config, job_name, env_vars)

        # Deploy the model to the created endpoint
        deploy_model_to_endpoint(model, endpoint, config)

    except Exception as e:
        logger.error(f"Run process failed: {e}")
        raise


if __name__ == "__main__":
    run()
