from src.models.gemini_1_5.evaluate import evaluate_model
from src.utils.plot import plot_metrics
from src.config.logging import logger
from src.config.loader import config
from google.cloud import aiplatform
from vertexai.tuning import sft
from typing import Optional
import os


# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')


def run(tuning_job_id: str) -> None:
    """
    Main function to run the evaluation pipeline for the Gemini 1.5 model.
    
    The process includes:
        1. Initializing the environment
        2. Retrieving the tuning job
        3. Evaluating the tuned model
        4. Plotting evaluation metrics

    Parameters:
        tuning_job_id (str): The ID of the tuning job to evaluate.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    logger.info("Starting the evaluation pipeline.")
    try:
        # Step 1: Initialize environment
        initialize_environment()

        # Step 2: Retrieve the tuning job
        tuning_job = get_tuning_job(tuning_job_id)

        # Step 3: Evaluate the tuned model
        evaluate_and_log_model(tuning_job)

        # Step 4: Plot evaluation metrics
        plot_and_log_metrics(tuning_job)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during the evaluation pipeline execution.")
        raise e


def initialize_environment() -> None:
    """
    Initializes the environment needed for the pipeline, including setting up Vertex AI configurations.

    Raises:
        Exception: If environment setup fails.
    """
    try:
        logger.info("Setting up the environment for Vertex AI.")
        project: Optional[str] = config.PROJECT.get('project_id')
        location: Optional[str] = config.PROJECT.get('location')
        
        if not project or not location:
            raise ValueError("Project ID or location is missing in the configuration.")
        
        aiplatform.init(project=project, location=location)
        logger.info("Environment setup successfully.")

    except Exception as e:
        logger.exception("Failed to setup the environment.")
        raise e


def get_tuning_job(tuning_job_id: str) -> sft.SupervisedTuningJob:
    """
    Retrieves the supervised tuning job using the provided ID.

    Parameters:
        tuning_job_id (str): The ID of the tuning job.

    Returns:
        sft.SupervisedTuningJob: The retrieved tuning job object.

    Raises:
        Exception: If retrieving the tuning job fails.
    """
    try:
        logger.info("Retrieving the tuning job with ID: %s", tuning_job_id)
        project: str = config.PROJECT.get('project_id')
        location: str = config.PROJECT.get('location')
        
        job = sft.SupervisedTuningJob(
            f"projects/{project}/locations/{location}/tuningJobs/{tuning_job_id}"
        )
        logger.info("Tuning job retrieved successfully.")
        return job

    except Exception as e:
        logger.exception("Failed to retrieve the tuning job with ID: %s", tuning_job_id)
        raise e


def evaluate_and_log_model(tuning_job: sft.SupervisedTuningJob) -> None:
    """
    Evaluates the tuned model and logs the evaluation process.

    Parameters:
        tuning_job (sft.SupervisedTuningJob): The object representing the tuning job.

    Raises:
        Exception: If the model evaluation fails.
    """
    try:
        logger.info("Evaluating the tuned model.")
        evaluate_model(tuning_job)
        logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.exception("Model evaluation failed.")
        raise e


def plot_and_log_metrics(tuning_job: sft.SupervisedTuningJob) -> None:
    """
    Plots the evaluation metrics and logs the process.

    Parameters:
        tuning_job (sft.SupervisedTuningJob): The object representing the tuning job.

    Raises:
        Exception: If metrics plotting fails.
    """
    try:
        logger.info("Plotting the evaluation metrics.")
        plot_metrics(tuning_job)
        logger.info("Metrics plotting completed successfully.")

    except Exception as e:
        logger.exception("Failed to plot metrics.")
        raise e


if __name__ == '__main__':
    tuning_job_id = "4577657336138563584"  # Replace with actual job ID
    run(tuning_job_id)
