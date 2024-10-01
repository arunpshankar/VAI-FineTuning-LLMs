from src.config.logging import logger
from typing import Dict
from typing import Any 
import subprocess
import json


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


if __name__ == "__main__":
    # Hardcoded values for testing
    accelerator_type = "NVIDIA_A100_80GB"  # You can change this to other types like "NVIDIA_TESLA_V100", "TPU_V3", etc.
    is_for_training = False  # True for training, False for serving
    is_restricted_image = True  # Set to True for restricted image testing
    is_dynamic_workload_scheduler = False  # Set to True if dynamic workload scheduler is being used

    # Fetch the resource ID
    resource_id = get_resource_id(
        accelerator_type=accelerator_type,
        is_for_training=is_for_training,
        is_restricted_image=is_restricted_image,
        is_dynamic_workload_scheduler=is_dynamic_workload_scheduler
    )

    # Output the result
    logger.info(f"Generated Resource ID: {resource_id}")