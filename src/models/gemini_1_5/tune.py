from vertexai.preview.tuning import sft
from src.config.logging import logger
from src.config.loader import config
from google.auth import default
import time
import os


# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')

def tune_model():
    """
    Tunes the Gemini model using supervised fine-tuning and logs the progress.

    Returns:
        sft_tuning_job: The completed tuning job object.
    """
    logger.info("Starting model tuning.")
    try:
        # Load hyperparameters and dataset paths from config
        tuned_model_display_name = config.HYPERPARAMETERS.get('tuned_model_display_name')
        epochs = config.HYPERPARAMETERS.get('epochs', 3)
        learning_rate_multiplier = config.HYPERPARAMETERS.get('learning_rate_multiplier', 1.0)
        adapter_size = config.HYPERPARAMETERS.get('adapter_size', 256)

        train_dataset = config.DATASET.get('train_dataset_path')
        validation_dataset = config.DATASET.get('validation_dataset_path')
        base_model = config.MODEL_CONFIG.get('base_model')

        # Validate configurations
        validate_tuning_parameters(
            tuned_model_display_name, epochs, learning_rate_multiplier, 
            adapter_size, train_dataset, validation_dataset, base_model
        )

        # Start the supervised fine-tuning job
        sft_tuning_job = start_tuning_job(
            base_model, train_dataset, validation_dataset, 
            epochs, learning_rate_multiplier, tuned_model_display_name, adapter_size
        )

        # Monitor the tuning job until completion
        monitor_tuning_job(sft_tuning_job)

        logger.info("Model tuning completed successfully.")
        return sft_tuning_job

    except Exception as e:
        logger.exception("An error occurred during model tuning.")
        raise e


def validate_tuning_parameters(tuned_model_display_name: str, epochs: int, learning_rate_multiplier: float, 
                               adapter_size: int, train_dataset: str, validation_dataset: str, base_model: str) -> None:
    """
    Validates that all required tuning parameters are set and logs the information.

    Parameters:
        tuned_model_display_name (str): Display name for the tuned model.
        epochs (int): Number of training epochs.
        learning_rate_multiplier (float): Multiplier for the learning rate.
        adapter_size (int): Size of the adapter used for fine-tuning.
        train_dataset (str): Path to the training dataset.
        validation_dataset (str): Path to the validation dataset.
        base_model (str): Base model to fine-tune.

    Raises:
        ValueError: If any required parameter is missing.
    """
    if not all([tuned_model_display_name, epochs, learning_rate_multiplier, adapter_size, train_dataset, validation_dataset, base_model]):
        logger.error("One or more tuning parameters are missing.")
        raise ValueError("Missing required tuning parameters. Please check the configuration.")

    logger.info(f"Hyperparameters: Epochs={epochs}, LearningRateMultiplier={learning_rate_multiplier}, AdapterSize={adapter_size}")
    logger.info(f"Datasets: TrainingData={train_dataset}, ValidationData={validation_dataset}")
    logger.info(f"BaseModel: {base_model}, TunedModelDisplayName: {tuned_model_display_name}")


def start_tuning_job(base_model: str, train_dataset: str, validation_dataset: str, epochs: int, 
                     learning_rate_multiplier: float, tuned_model_display_name: str, adapter_size: int):
    """
    Starts the supervised fine-tuning job.

    Parameters:
        base_model (str): The base model to fine-tune.
        train_dataset (str): The path to the training dataset.
        validation_dataset (str): The path to the validation dataset.
        epochs (int): Number of epochs for training.
        learning_rate_multiplier (float): The learning rate multiplier.
        tuned_model_display_name (str): The display name for the tuned model.
        adapter_size (int): Adapter size for the model fine-tuning.

    Returns:
        sft_tuning_job: The tuning job object.
    """
    try:
        logger.info(f"Starting tuning job with model {base_model} and display name {tuned_model_display_name}.")

        return sft.train(
            source_model=base_model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            learning_rate_multiplier=learning_rate_multiplier,
            tuned_model_display_name=tuned_model_display_name,
            adapter_size=adapter_size
        )
    except Exception as e:
        logger.exception("Failed to start the tuning job.")
        raise e


def monitor_tuning_job(sft_tuning_job) -> None:
    """
    Monitors the progress of the supervised fine-tuning job until it completes.

    Parameters:
        sft_tuning_job: The tuning job object to monitor.
    """
    try:
        while not sft_tuning_job.refresh().has_ended:
            logger.info("Tuning job in progress...")
            time.sleep(60)  # Sleep for a minute before checking the status again
    except Exception as e:
        logger.exception("An error occurred while monitoring the tuning job.")
        raise e
