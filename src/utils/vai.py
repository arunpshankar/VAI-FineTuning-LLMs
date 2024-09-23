from src.config.logging import logger 
from datetime import datetime
import subprocess
import json


def get_job_name_with_datetime(prefix: str) -> str:
    """Generates a unique job name with a timestamp.

    Args:
        prefix (str): The prefix for the job name.

    Returns:
        str: A unique job name in the format 'prefix-YYYYMMDDHHMMSS'.
    """
    job_name = f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.debug(f"Generated job name: {job_name}")
    return job_name


def get_quota(project_id: str, region: str, resource_id: str) -> int:
    """Returns the quota for a resource in a specified region.

    Args:
        project_id (str): The Google Cloud project ID.
        region (str): The region to check the quota in.
        resource_id (str): The resource ID to check quota for.

    Returns:
        int: The quota for the resource in the region.
             Returns -1 if unable to determine the quota.

    Raises:
        RuntimeError: If the command to get quota fails.
    """
    service_endpoint = "aiplatform.googleapis.com"
    command = (
        f"gcloud alpha services quota list"
        f" --service={service_endpoint}"
        f" --consumer=projects/{project_id}"
        f" --filter='{service_endpoint}/{resource_id}'"
        f" --format=json"
    )

    try:
        logger.debug(f"Running command to get quota: {command}")
        process = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        quota_data = json.loads(process.stdout)
        logger.debug(f"Quota data retrieved: {quota_data}")
    except subprocess.CalledProcessError as e:
        logger.exception("Failed to fetch quota data.")
        raise RuntimeError(f"Error fetching quota data: {e.stderr}") from e

    try:
        if not quota_data or "consumerQuotaLimits" not in quota_data[0]:
            logger.warning("Quota data is empty or missing 'consumerQuotaLimits'.")
            return -1
        quota_limits = quota_data[0]["consumerQuotaLimits"]
        if not quota_limits or "quotaBuckets" not in quota_limits[0]:
            logger.warning("Quota limits are empty or missing 'quotaBuckets'.")
            return -1
        quota_buckets = quota_limits[0]["quotaBuckets"]
        for region_data in quota_buckets:
            dimensions = region_data.get("dimensions", {})
            if dimensions.get("region") == region:
                effective_limit = region_data.get("effectiveLimit")
                if effective_limit is not None:
                    logger.info(f"Quota for {resource_id} in {region}: {effective_limit}")
                    return int(effective_limit)
                else:
                    logger.warning(f"No 'effectiveLimit' found for {resource_id} in {region}.")
                    return 0
        logger.warning(f"No quota data found for region: {region}.")
        return -1
    except (IndexError, KeyError, ValueError, TypeError) as e:
        logger.exception("Error parsing quota data.")
        raise RuntimeError("Error parsing quota data.") from e


def get_resource_id(
    accelerator_type: str,
    is_for_training: bool,
    is_restricted_image: bool = False,
) -> str:
    """Returns the resource ID for a given accelerator type and use case.

    Args:
        accelerator_type (str): The accelerator type (e.g., 'NVIDIA_L4').
        is_for_training (bool): True if the resource is used for training; False for serving.
        is_restricted_image (bool, optional): True if using a restricted image. Defaults to False.

    Returns:
        str: The resource ID corresponding to the accelerator type and use case.

    Raises:
        ValueError: If the accelerator type is not recognized.
    """
    default_training_accelerator_map = {
        "NVIDIA_TESLA_V100": "custom_model_training_nvidia_v100_gpus",
        "NVIDIA_L4": "custom_model_training_nvidia_l4_gpus",
        "NVIDIA_TESLA_A100": "custom_model_training_nvidia_a100_gpus",
        "NVIDIA_A100_80GB": "custom_model_training_nvidia_a100_80gb_gpus",
        "NVIDIA_H100_80GB": "custom_model_training_nvidia_h100_gpus",
        "NVIDIA_TESLA_T4": "custom_model_training_nvidia_t4_gpus",
        "TPU_V5e": "custom_model_training_tpu_v5e",
        "TPU_V3": "custom_model_training_tpu_v3",
    }
    restricted_image_training_accelerator_map = {
        "NVIDIA_A100_80GB": "restricted_image_training_nvidia_a100_80gb_gpus",
    }
    serving_accelerator_map = {
        "NVIDIA_TESLA_V100": "custom_model_serving_nvidia_v100_gpus",
        "NVIDIA_L4": "custom_model_serving_nvidia_l4_gpus",
        "NVIDIA_TESLA_A100": "custom_model_serving_nvidia_a100_gpus",
        "NVIDIA_A100_80GB": "custom_model_serving_nvidia_a100_80gb_gpus",
        "NVIDIA_H100_80GB": "custom_model_serving_nvidia_h100_gpus",
        "NVIDIA_TESLA_T4": "custom_model_serving_nvidia_t4_gpus",
        "TPU_V5e": "custom_model_serving_tpu_v5e",
    }

    if is_for_training:
        accelerator_map = (
            restricted_image_training_accelerator_map
            if is_restricted_image
            else default_training_accelerator_map
        )
        use_case = "training"
    else:
        accelerator_map = serving_accelerator_map
        use_case = "serving"

    resource_id = accelerator_map.get(accelerator_type)
    if resource_id:
        logger.debug(f"Resource ID for {accelerator_type} ({use_case}): {resource_id}")
        return resource_id
    else:
        error_msg = f"Could not find accelerator type: {accelerator_type} for {use_case}."
        logger.error(error_msg)
        raise ValueError(error_msg)


def check_quota(
    project_id: str,
    region: str,
    accelerator_type: str,
    accelerator_count: int,
    is_for_training: bool,
    is_restricted_image: bool = False,
) -> None:
    """Checks if the project and region have the required quota.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Region to check quota in.
        accelerator_type (str): Accelerator type (e.g., 'NVIDIA_L4').
        accelerator_count (int): Number of accelerators required.
        is_for_training (bool): True if checking quota for training resources.
        is_restricted_image (bool, optional): True if using a restricted image. Defaults to False.

    Raises:
        ValueError: If quota is insufficient or cannot be determined.
    """
    logger.info("Checking resource quotas.")
    try:
        resource_id = get_resource_id(
            accelerator_type, is_for_training, is_restricted_image
        )
        quota = get_quota(project_id, region, resource_id)
    except Exception as e:
        logger.exception("Failed to check resource quotas.")
        raise e

    quota_request_instruction = (
        "Either use a different region or request additional quota. Follow instructions here: "
        "https://cloud.google.com/docs/quotas#requesting_higher_quota "
        "to check quota in a region or request additional quota for your project."
    )

    if quota == -1:
        error_msg = f"Quota not found for: {resource_id} in {region}. {quota_request_instruction}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif quota < accelerator_count:
        error_msg = (
            f"Quota not sufficient for {resource_id} in {region}: "
            f"available {quota} < required {accelerator_count}. {quota_request_instruction}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        logger.info(f"Resource quota is sufficient: {quota} >= {accelerator_count}")
