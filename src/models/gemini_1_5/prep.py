import logging
from google.cloud import storage
import os
from typing import List, Dict
import jsonlines
from src.config.config import config  # Import the Config singleton

def prepare_data():
    """Prepares the dataset for model tuning."""
    logging.info("Starting data preparation.")
    try:
        # Access dataset paths from Config
        train_file_local = 'sft_train_samples.jsonl'
        val_file_local = 'sft_val_samples.jsonl'

        train_dataset_path = config.DATASET.get('train_dataset_path')
        val_dataset_path = config.DATASET.get('validation_dataset_path')

        # Convert datasets
        train_instances = create_tuning_samples(train_file_local)
        save_jsonlines(train_file_local, train_instances)

        val_instances = create_tuning_samples(val_file_local)
        save_jsonlines(val_file_local, val_instances)

        # Upload to GCS
        upload_to_gcs(train_file_local, train_dataset_path)
        upload_to_gcs(val_file_local, val_dataset_path)

        logging.info("Data preparation completed successfully.")
    except Exception as e:
        logging.exception("An error occurred during data preparation.")
        raise e

def create_tuning_samples(file_path: str) -> List[Dict]:
    """Creates tuning samples from a file."""
    try:
        with jsonlines.open(file_path) as reader:
            instances = []
            for obj in reader:
                instance = []
                for content in obj["messages"]:
                    instance.append(
                        {"role": content["role"], "parts": [{"text": content["content"]}]}
                    )
                instances.append({"contents": instance})
        return instances
    except Exception as e:
        logging.exception(f"Failed to create tuning samples from {file_path}.")
        raise e

def save_jsonlines(file: str, instances: List[Dict]) -> None:
    """Saves a list of json instances to a jsonlines file."""
    try:
        with jsonlines.open(file, mode="w") as writer:
            writer.write_all(instances)
    except Exception as e:
        logging.exception(f"Failed to save jsonlines to {file}.")
        raise e

def upload_to_gcs(local_file: str, gcs_uri: str) -> None:
    """Uploads a local file to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket_name = gcs_uri.split('/')[2]
        destination_blob_name = '/'.join(gcs_uri.split('/')[3:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file)
        logging.info(f"Uploaded {local_file} to {gcs_uri}.")
    except Exception as e:
        logging.exception(f"Failed to upload {local_file} to {gcs_uri}.")
        raise e
