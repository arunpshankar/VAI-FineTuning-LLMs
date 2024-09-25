from src.config.logging import logger
from src.config.loader import config
from google.cloud import storage
from typing import Dict
from typing import List
from typing import Any
import jsonlines


def prepare_data() -> None:
    """
    Prepares the dataset for model fine-tuning by creating tuning samples from
    local JSONL files, saving them, and uploading them to Google Cloud Storage.

    Raises:
        Exception: If any step in the data preparation process fails.
    """
    logger.info("Starting data preparation.")
    try:
        # Access dataset paths from the config
        train_file_local = './data/gemini_1_5/sft_train_samples.jsonl'
        val_file_local = './data/gemini_1_5/sft_val_samples.jsonl'

        train_dataset_path = config.DATASET.get('train_dataset_path')
        val_dataset_path = config.DATASET.get('validation_dataset_path')

        # Create tuning samples and save them to JSONL
        train_instances = create_tuning_samples(train_file_local)
        save_jsonlines(train_file_local, train_instances)

        val_instances = create_tuning_samples(val_file_local)
        save_jsonlines(val_file_local, val_instances)

        # Upload files to Google Cloud Storage (GCS)
        upload_to_gcs(train_file_local, train_dataset_path)
        upload_to_gcs(val_file_local, val_dataset_path)

        logger.info("Data preparation completed successfully.")
    except Exception as e:
        logger.exception("An error occurred during data preparation.")
        raise e


def create_tuning_samples(file_path: str) -> List[Dict[str, Any]]:
    """
    Creates tuning samples by reading a JSONL file and transforming the content
    into a format suitable for model fine-tuning.

    Args:
        file_path (str): The path to the input JSONL file containing message data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the tuning samples.

    Raises:
        Exception: If there's an error during file reading or processing.
    """
    try:
        instances: List[Dict[str, Any]] = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                instance = []
                for content in obj["messages"]:
                    instance.append(
                        {"role": content["role"], "parts": [{"text": content["content"]}]}
                    )
                instances.append({"contents": instance})
        return instances
    except Exception as e:
        logger.exception(f"Failed to create tuning samples from {file_path}.")
        raise e


def save_jsonlines(file: str, instances: List[Dict[str, Any]]) -> None:
    """
    Saves a list of JSON objects to a JSONL file.

    Args:
        file (str): The path to the output JSONL file.
        instances (List[Dict[str, Any]]): A list of dictionaries to write to the file.

    Raises:
        Exception: If there's an error during the file writing process.
    """
    try:
        with jsonlines.open(file, mode="w") as writer:
            writer.write_all(instances)
        logger.info(f"Saved tuning samples to {file}.")
    except Exception as e:
        logger.exception(f"Failed to save jsonlines to {file}.")
        raise e


def upload_to_gcs(local_file: str, gcs_uri: str) -> None:
    """
    Uploads a local file to Google Cloud Storage (GCS) based on the provided GCS URI.

    Args:
        local_file (str): The path to the local file to upload.
        gcs_uri (str): The destination GCS URI in the format 'gs://bucket_name/path/to/file'.

    Raises:
        Exception: If there's an error during the file upload process.
    """
    try:
        client = storage.Client()
        bucket_name = gcs_uri.split('/')[2]
        destination_blob_name = '/'.join(gcs_uri.split('/')[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file)
        logger.info(f"Uploaded {local_file} to {gcs_uri}.")
    except Exception as e:
        logger.exception(f"Failed to upload {local_file} to {gcs_uri}.")
        raise e
