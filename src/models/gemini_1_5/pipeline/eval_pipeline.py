from src.models.gemini_1_5.evaluate import evaluate_model
from src.utils.plot import plot_metrics
from src.config.logging import logger
from src.config.loader import config
from google.cloud import aiplatform
from vertexai.tuning import sft
import os


# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')

def run():
    """
    Main function to run the evaluation pipeline for Gemini 1.5 model.
    This function follows these steps:
        1. Initialize environment
        2. Get the tuning job
        3. Evaluate the tuned model
        4. Plot evaluation metrics
    """
    logger.info("Starting the evaluation pipeline.")
    try:
        # Step 1: Initialize environment
        initialize_environment()

        # Step 2: Get the tuning job
        tuning_job = get_tuning_job()

        # Step 3: Model evaluation
        evaluate_and_log_model(tuning_job)

        # Step 4: Plot evaluation metrics
        plot_and_log_metrics(tuning_job)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during the evaluation pipeline execution.")
        raise e


def initialize_environment() -> None:
    """
    Initializes the environment needed for the pipeline.
    This includes setting up Vertex AI configurations.
    """
    try:
        logger.info("Setting up the environment for Vertex AI.")
        project = config.PROJECT.get('project_id')
        location = config.PROJECT.get('location')
        aiplatform.init(project=project, location=location)
    except Exception as e:
        logger.exception("Failed to setup the environment.")
        raise e


def get_tuning_job():
    """
    Retrieves the tuning job using its ID.

    Returns:
        The SupervisedTuningJob object.
    """
    try:
        logger.info("Retrieving the tuning job.")
        project = config.PROJECT.get('project_id')
        location = config.PROJECT.get('location')
        tuning_job_id = "40843661516210176"  # You might want to make this configurable
        job = sft.SupervisedTuningJob(
            f"projects/{project}/locations/{location}/tuningJobs/{tuning_job_id}"
        )
        return job
    except Exception as e:
        logger.exception("Failed to retrieve the tuning job.")
        raise e


def evaluate_and_log_model(tuning_job) -> None:
    """
    Evaluates the tuned model and logs the evaluation process.

    Parameters:
        tuning_job: The object representing the tuning job.
    """
    try:
        logger.info("Evaluating the tuned model.")
        evaluate_model(tuning_job)
        logger.info("Model evaluation completed.")
    except Exception as e:
        logger.exception("Model evaluation failed.")
        raise e


def plot_and_log_metrics(tuning_job) -> None:
    """
    Plots the evaluation metrics and logs the process.

    Parameters:
        tuning_job: The object representing the tuning job.
    """
    try:
        logger.info("Plotting the evaluation metrics.")
        plot_metrics(tuning_job)
        logger.info("Metrics plotting completed.")
    except Exception as e:
        logger.exception("Failed to plot metrics.")
        raise e


if __name__ == '__main__':
    run()