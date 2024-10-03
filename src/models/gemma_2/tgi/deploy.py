from src.utils.common import get_job_name_with_datetime
from google.cloud.aiplatform import Endpoint, Model
from src.utils.quota import check_quota
from src.config.logging import logger
from src.config.loader import Config 
from typing import Tuple, Dict, Any





config = Config(model_name="gemma_2", reinitialize=True)

# Dictionary to store deployed models and endpoints
models: Dict[str, Model] = {}
endpoints: Dict[str, Endpoint] = {}


def deploy_model_tgi(model_name: str) -> Tuple[Model, Endpoint]:
    """
    Deploys a model with TGI (Text Generation Inference) on GPU in Vertex AI.

    Args:
        model_name (str): The name of the model to be deployed.
        service_account (str): The service account to use for deploying the model.

    Returns:
        Tuple[Model, Endpoint]: The deployed model and the corresponding endpoint.

    Raises:
        ValueError: If the deployment or upload process fails.
        Exception: For any other unexpected errors.
    """
    try:
        logger.info(f"Starting deployment for model: {model_name}")

        # Creating the endpoint
        endpoint = Endpoint.create(display_name=f"{model_name}-endpoint", 
                                   dedicated_endpoint_enabled=config.DEPLOYMENT.get('use_dedicated_endpoint'))
        logger.info(f"Endpoint created with name: {endpoint.display_name}")

        # Setting environment variables for the deployment
        env_vars = {
            "MODEL_ID": config.MODEL.get('model_id'),
            "NUM_SHARD": config.DEPLOYMENT.get('accelerator_count'),
            "MAX_INPUT_LENGTH": config.MODEL.get('max_input_length'),
            "MAX_TOTAL_TOKENS": config.MODEL.get('max_total_tokens'),
            "MAX_BATCH_PREFILL_TOKENS": config.MODEL.get('max_batch_prefill_tokens'),
            "DEPLOY_SOURCE": "notebook",
        }

        logger.info(f"Uploading model: {model_name} with environment vars: {env_vars}")

        # Uploading the model to Vertex AI
        model = Model.upload(
            display_name=model_name,
            serving_container_image_uri=config.MODEL.get('TGI_DOCKER_URI'),
            serving_container_ports=[8080],
            serving_container_environment_variables=env_vars,
            serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB shared memory
        )
        logger.info(f"Model {model_name} uploaded successfully")

        # Deploying the model to the created endpoint
        model.deploy(
            endpoint=endpoint,
            machine_type=config.DEPLOYMENT.get('machine_type'),
            accelerator_type=config.DEPLOYMENT.get('accelerator_type'),
            accelerator_count=config.DEPLOYMENT.get('accelerator_count'),
            deploy_request_timeout=1800,  # 30 minutes timeout
            #service_account=service_account
        )

        logger.info(f"Model {model_name} deployed successfully to endpoint: {endpoint.display_name}")
        return model, endpoint

    except ValueError as ve:
        logger.error(f"ValueError during deployment of {model_name}: {ve}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during deployment: {str(e)}")
        raise


if __name__ == '__main__':
    models["tgi"], endpoints["tgi"] = deploy_model_tgi(model_name=get_job_name_with_datetime(prefix=config.MODEL.get('MODEL_ID')))
