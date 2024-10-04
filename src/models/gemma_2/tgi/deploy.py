from src.utils.common import get_job_name_with_datetime, load_yaml
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform import Model 
from src.utils.quota import check_quota
from src.config.logging import logger
from src.config.loader import Config 
from typing import Tuple
from typing import Dict 
from typing import Any 
import os 


config = Config(model_name="gemma_2", reinitialize=True)
# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')


logger.info(config.MODEL)
logger.info(config.DEPLOYMENT)

model_name = config.MODEL.get('model_id')


job_name = get_job_name_with_datetime(prefix=model_name)
print(job_name)


endpoint = Endpoint.create(display_name=f"{job_name}_endpoint", 
                           dedicated_endpoint_enabled=config.DEPLOYMENT.get('use_dedicated_endpoint'))



 # Setting environment variables for the deployment
env_vars = {
    "MODEL_ID": config.MODEL.get('model_id'),
    "NUM_SHARD": config.DEPLOYMENT.get('accelerator_count'),
    "MAX_INPUT_LENGTH": config.MODEL.get('max_input_length'),
    "MAX_TOTAL_TOKENS": config.MODEL.get('max_total_tokens'),
    "MAX_BATCH_PREFILL_TOKENS": config.MODEL.get('max_batch_prefill_tokens'),
    "DEPLOY_SOURCE": "VERTEX_AI_SDK"
    
}

logger.info(f"Uploading model: {model_name} with environment vars: {env_vars}")


def get_key(yaml_data, key):
    return yaml_data.get(key, f"{key} not found in YAML file")

# Load the YAML file
file_path = './credentials/hf.yml'
yaml_data = load_yaml(file_path)
hf_token = get_key(yaml_data, 'key')

env_vars["HF_TOKEN"] = hf_token


print(env_vars)


# Uploading the model to Vertex AI
model = Model.upload(
    display_name=job_name,
    serving_container_image_uri=config.MODEL.get('TGI_DOCKER_URI'),
    serving_container_ports=[8080],
    serving_container_environment_variables=env_vars,
    serving_container_shared_memory_size_mb=(16 * 1024),  # 16 GB shared memory
)
logger.info(f"Model {job_name} uploaded successfully")


# Deploying the model to the created endpoint
model.deploy(
    endpoint=endpoint,
    machine_type=config.DEPLOYMENT.get('machine_type'),
    accelerator_type=config.DEPLOYMENT.get('accelerator_type'),
    accelerator_count=config.DEPLOYMENT.get('accelerator_count'),
    deploy_request_timeout=1800,  # 30 minutes timeout
    service_account='as-service@arun-genai-bb.iam.gserviceaccount.com'
)
