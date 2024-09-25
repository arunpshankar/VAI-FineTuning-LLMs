from src.models.gemini_1_5.evaluate import evaluate_model
from src.models.gemini_1_5.prep import prepare_data
from src.models.gemini_1_5.tune import tune_model
from src.setup import setup_environment
from src.config.logging import logger
from src.utils.plot import plot_metrics


def run():
    """
    Main function to run the supervised fine-tuning pipeline for Gemini 1.5 model.
    This function follows these steps:
        1. Setup environment
        2. Prepare data
        3. Tune the model
    """
    logger.info("Starting the supervised fine-tuning pipeline.")
    try:
        # Step 1: Initialize Vertex AI environment
        initialize_environment()

        # Step 2: Data preparation
        prepare_and_log_data()

        # Step 3: Model tuning
        tuning_job = tune_and_log_model()
        print(tuning_job)
        print(tuning_job.__dict__)

        logger.info("Pipeline completed successfully.")
    
    except Exception as e:
        logger.exception("An error occurred during the fine-tuning pipeline execution.")
        raise e


def initialize_environment() -> None:
    """
    Initializes the environment needed for the pipeline.
    This includes setting up Vertex AI configurations.
    """
    try:
        logger.info("Setting up the environment for Vertex AI.")
        setup_environment()
    except Exception as e:
        logger.exception("Failed to setup the environment.")
        raise e


def prepare_and_log_data() -> None:
    """
    Prepares the data for model fine-tuning and logs the process.
    """
    try:
        logger.info("Preparing the data for fine-tuning.")
        prepare_data()
    except Exception as e:
        logger.exception("Data preparation failed.")
        raise e


def tune_and_log_model():
    """
    Tunes the model using supervised fine-tuning and logs the process.

    Returns:
        tuning_job: The object representing the tuning job.
    """
    try:
        logger.info("Tuning the model.")
        tuning_job = tune_model()
        logger.info("Model tuning completed.")
        return tuning_job
    except Exception as e:
        logger.exception("Model tuning failed.")
        raise e


if __name__ == '__main__':
    run()
