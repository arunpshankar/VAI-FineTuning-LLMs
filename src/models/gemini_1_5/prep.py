from google.oauth2 import service_account
from src.config.logging import logger
from src.config.loader import config
from google.cloud import storage
import os 


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
        train_file_local = config.DATASET.get('train_dataset_local_path')
        val_file_local = config.DATASET.get('validation_dataset_local_path')

        train_dataset_path = config.DATASET.get('train_dataset_path')
        val_dataset_path = config.DATASET.get('validation_dataset_path')

        # Upload files to Google Cloud Storage (GCS)
        upload_to_gcs(train_file_local, train_dataset_path)
        upload_to_gcs(val_file_local, val_dataset_path)

        logger.info("Data preparation completed successfully.")
    except Exception as e:
        logger.exception("An error occurred during data preparation.")
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
        # Get the path to the service account key file from your config
        key_path = config.PROJECT.get('credentials_path')
        
        if not key_path or not os.path.exists(key_path):
            raise FileNotFoundError(f"Service account key file not found at {key_path}")

        # Create credentials using the service account key file
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Initialize the storage client with the credentials
        client = storage.Client(credentials=credentials, project=credentials.project_id)
        
        bucket_name = gcs_uri.split('/')[2]
        destination_blob_name = '/'.join(gcs_uri.split('/')[3:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_filename(local_file)
        
        logger.info(f"Uploaded {local_file} to {gcs_uri}.")
    except FileNotFoundError as e:
        logger.error(f"Service account key file not found: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Failed to upload {local_file} to {gcs_uri}.")
        raise e
