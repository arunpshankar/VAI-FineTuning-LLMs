from src.config.logging import logger
from src.config.loader import Config
from typing import Optional
import subprocess
import json 
import os 


# Load default config
config = Config(model_name=None)

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')

# Mapping accelerator types to their respective resource suffixes
ACCELERATOR_SUFFIX_MAP = {
    "NVIDIA_TESLA_V100": "nvidia_v100_gpus",
    "NVIDIA_L4": "nvidia_l4_gpus",
    "NVIDIA_TESLA_A100": "nvidia_a100_gpus",
    "NVIDIA_A100_80GB": "nvidia_a100_80gb_gpus",
    "NVIDIA_H100_80GB": "nvidia_h100_gpus",
    "NVIDIA_TESLA_T4": "nvidia_t4_gpus",
    "TPU_V5e": "tpu_v5e",
    "TPU_V3": "tpu_v3",
}


def get_resource_id(
    accelerator_type: str,
    is_for_training: bool,
    is_restricted_image: bool = False,
    is_dynamic_workload_scheduler: bool = False
) -> str:
    """
    Returns the resource ID for a given accelerator type and use case.

    Args:
        accelerator_type (str): The accelerator type.
        is_for_training (bool): Whether the resource is for training or serving.
        is_restricted_image (bool, optional): Whether the image is hosted in a restricted environment. Defaults to False.
        is_dynamic_workload_scheduler (bool, optional): Whether the resource is used with Dynamic Workload Scheduler. Defaults to False.

    Returns:
        str: The resource ID.

    Raises:
        ValueError: If incompatible combinations of parameters are provided.
    """
    logger.info(f"Received request for resource ID: {accelerator_type}, "
                f"is_for_training={is_for_training}, is_restricted_image={is_restricted_image}, "
                f"is_dynamic_workload_scheduler={is_dynamic_workload_scheduler}")
    try:
        if is_for_training:
            if is_restricted_image and is_dynamic_workload_scheduler:
                logger.error("Dynamic Workload Scheduler does not work for restricted image training.")
                raise ValueError("Dynamic Workload Scheduler does not work for restricted image training.")
            return _get_training_id(accelerator_type, is_restricted_image, is_dynamic_workload_scheduler)
        else:
            if is_dynamic_workload_scheduler:
                logger.error("Dynamic Workload Scheduler does not work for serving.")
                raise ValueError("Dynamic Workload Scheduler does not work for serving.")
            return _get_serving_id(accelerator_type)
    except ValueError as e:
        logger.error(f"Error in get_resource_id: {str(e)}")
        raise


def _get_accelerator_suffix(accelerator_type: str) -> str:
    """
    Get the corresponding resource suffix for the given accelerator type.

    Args:
        accelerator_type (str): The type of accelerator.

    Returns:
        str: The resource suffix associated with the accelerator type.

    Raises:
        ValueError: If the accelerator type is not supported.
    """
    if accelerator_type not in ACCELERATOR_SUFFIX_MAP:
        logger.error(f"Unsupported accelerator type: {accelerator_type}")
        raise ValueError(f"Unsupported accelerator type: {accelerator_type}")
    logger.info(f"Retrieved suffix for accelerator type: {accelerator_type}")
    return ACCELERATOR_SUFFIX_MAP[accelerator_type]


def _get_training_prefix(is_dynamic_workload_scheduler: bool) -> str:
    """
    Returns the appropriate prefix based on whether a dynamic workload scheduler is used.

    Args:
        is_dynamic_workload_scheduler (bool): Indicates if a dynamic workload scheduler is being used.

    Returns:
        str: The training prefix.
    """
    prefix = "custom_model_training_preemptible" if is_dynamic_workload_scheduler else "custom_model_training"
    logger.info(f"Determined training prefix: {prefix}")
    return prefix


def _get_restricted_training_id(accelerator_type: str) -> str:
    """
    Returns the restricted image training ID for specific accelerator types.

    Args:
        accelerator_type (str): The type of accelerator.

    Returns:
        str: The restricted image training ID.

    Raises:
        ValueError: If the accelerator type is unsupported for restricted images.
    """
    if accelerator_type != "NVIDIA_A100_80GB":
        logger.error(f"Unsupported accelerator for restricted image: {accelerator_type}")
        raise ValueError(f"Unsupported accelerator for restricted image: {accelerator_type}")
    logger.info("Retrieved restricted image training ID for NVIDIA_A100_80GB")
    return "restricted_image_training_nvidia_a100_80gb_gpus"


def _get_training_id(accelerator_type: str, is_restricted_image: bool, is_dynamic_workload_scheduler: bool) -> str:
    """
    Returns the appropriate training resource ID based on accelerator type and other conditions.

    Args:
        accelerator_type (str): The type of accelerator.
        is_restricted_image (bool): Whether the image is restricted.
        is_dynamic_workload_scheduler (bool): Whether a dynamic workload scheduler is being used.

    Returns:
        str: The training resource ID.
    """
    if is_restricted_image:
        logger.info(f"Using restricted image for accelerator type: {accelerator_type}")
        return _get_restricted_training_id(accelerator_type)

    suffix = _get_accelerator_suffix(accelerator_type)
    prefix = _get_training_prefix(is_dynamic_workload_scheduler)
    training_id = f"{prefix}_{suffix}"
    logger.info(f"Generated training resource ID: {training_id}")
    return training_id


def _get_serving_id(accelerator_type: str) -> str:
    """
    Returns the serving resource ID for the given accelerator type.

    Args:
        accelerator_type (str): The type of accelerator.

    Returns:
        str: The serving resource ID.
    """
    suffix = _get_accelerator_suffix(accelerator_type)
    serving_id = f"custom_model_serving_{suffix}"
    logger.info(f"Generated serving resource ID: {serving_id}")
    return serving_id


def get_quota(project_id: str, region: str, resource_id: str) -> int:
    """
    Returns the quota for a resource in a specific region.

    Args:
        project_id: The project id.
        region: The region.
        resource_id: The resource id.

    Returns:
        The quota for the resource in the region. Returns -1 if unable to determine the quota.
    """
    try:
        quota_data = _fetch_quota_data(project_id, resource_id)
        return _extract_quota_for_region(quota_data, region)
    except Exception as e:
        logger.error(f"Failed to get quota: {e}")
        return -1

def _fetch_quota_data(project_id: str, resource_id: str) -> Optional[dict]:
    """
    Fetches quota data from the gcloud command.

    Args:
        project_id: The project id.
        resource_id: The resource id.

    Returns:
        The quota data in JSON format as a dictionary.
    
    Raises:
        RuntimeError: If the gcloud command to fetch quota data fails.
    """
    service_endpoint = "aiplatform.googleapis.com"
    command = (
        f"gcloud alpha services quota list --service={service_endpoint} "
        f"--consumer=projects/{project_id} "
        f"--filter='{service_endpoint}/{resource_id}' --format=json"
    )
    
    try:
        process = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        logger.info(f"Quota data fetched successfully for project {project_id}.")
        return json.loads(process.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing gcloud command: {e.stderr}")
        raise RuntimeError(f"Error fetching quota data: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        raise RuntimeError(f"Error decoding quota data: {e}")
    

def _extract_quota_for_region(quota_data: Optional[dict], region: str) -> int:
    """Extracts the effective quota limit for a specific region from the quota data.

    Args:
        quota_data: The quota data dictionary.
        region: The region for which to extract the quota.

    Returns:
        The effective quota limit for the region, or -1 if not found.
    """
    if not quota_data or "consumerQuotaLimits" not in quota_data[0]:
        logger.warning("Invalid or empty quota data.")
        return -1
    
    quota_limits = quota_data[0].get("consumerQuotaLimits", [])
    if not quota_limits or "quotaBuckets" not in quota_limits[0]:
        logger.warning("Quota limits or quota buckets not found.")
        return -1
    
    for region_data in quota_limits[0]["quotaBuckets"]:
        if region_data.get("dimensions", {}).get("region") == region:
            effective_limit = region_data.get("effectiveLimit")
            if effective_limit is not None:
                logger.info(f"Quota for region {region}: {effective_limit}")
                return int(effective_limit)
            else:
                logger.info(f"No effective limit found for region {region}.")
                return 0
    
    logger.warning(f"No data found for region {region}.")
    return -1


def check_quota(
    project_id: str,
    region: str,
    accelerator_type: str,
    accelerator_count: int,
    is_for_training: bool,
    is_restricted_image: bool = False,
    is_dynamic_workload_scheduler: bool = False,
) -> None:
    """
    Checks if the project and the region have the required quota for a specific resource.
    
    Args:
        project_id (str): The project ID.
        region (str): The region where the quota check is required.
        accelerator_type (str): The type of accelerator.
        accelerator_count (int): The number of accelerators needed.
        is_for_training (bool): Flag indicating if the resource is for training.
        is_restricted_image (bool, optional): Flag indicating if the resource is using a restricted image. Defaults to False.
        is_dynamic_workload_scheduler (bool, optional): Flag indicating if a dynamic workload scheduler is used. Defaults to False.

    Raises:
        ValueError: If the quota is not found or is insufficient for the request.
    """
    
    logger.info("Checking quota for project: %s in region: %s", project_id, region)
    
    try:
        resource_id: str = get_resource_id(
            accelerator_type,
            is_for_training=is_for_training,
            is_restricted_image=is_restricted_image,
            is_dynamic_workload_scheduler=is_dynamic_workload_scheduler,
        )
        logger.info("Resource ID determined: %s", resource_id)

        quota: Optional[int] = get_quota(project_id, region, resource_id)
        quota_request_instruction = (
            "Either use a different region or request additional quota. "
            "Follow instructions here: "
            "https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota"
            " to check quota in a region or request additional quota for your project."
        )

        if quota == -1:
            logger.error("Quota not found for resource ID: %s in region: %s", resource_id, region)
            raise ValueError(
                f"Quota not found for: {resource_id} in {region}. {quota_request_instruction}"
            )

        if quota < accelerator_count:
            logger.error("Insufficient quota for resource ID: %s in region: %s. Available: %d, Required: %d", 
                         resource_id, region, quota, accelerator_count)
            raise ValueError(
                f"Quota not enough for {resource_id} in {region}: {quota} < {accelerator_count}. "
                f"{quota_request_instruction}"
            )

        logger.info("Sufficient quota available for resource ID: %s in region: %s", resource_id, region)

    except ValueError as e:
        logger.error("Error during quota check: %s", str(e))
        raise


if __name__ == "__main__":
    accelerator_type = "NVIDIA_A100_80GB"  # You can change this to other types like "NVIDIA_TESLA_V100", "TPU_V3", etc.
    accelerator_count = 1
    is_for_training = False  # True for training, False for serving
    is_restricted_image = True  # Set to True for restricted image testing
    is_dynamic_workload_scheduler = False  # Set to True if dynamic workload scheduler is being used

    project_id = config.PROJECT.get('project_id')
    region = config.PROJECT.get('location')

    # Fetch the resource ID
    resource_id = get_resource_id(
        accelerator_type=accelerator_type,
        is_for_training=is_for_training,
        is_restricted_image=is_restricted_image,
        is_dynamic_workload_scheduler=is_dynamic_workload_scheduler
    )

    logger.info(f"Generated Resource ID: {resource_id}")

    quota = get_quota(project_id, region, resource_id)

    if quota != -1:
        logger.info(f"Quota for {resource_id} in region {region}: {quota}")
    else:
        logger.error(f"Failed to retrieve quota for {resource_id} in region {region}")


    check_quota(project_id, region, accelerator_type, accelerator_count, is_for_training, is_restricted_image, is_dynamic_workload_scheduler)