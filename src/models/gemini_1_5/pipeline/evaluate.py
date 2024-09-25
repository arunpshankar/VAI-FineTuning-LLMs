from src.config.logging import logger 
from src.config.loader import config
from google.cloud import aiplatform
from vertexai.tuning import sft
import os 


# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')


def get_job_using_id(
    project: str,
    location: str,
    tuning_job_id: str,
    
):
    """
    Recreates a Vertex AI hyperparameter tuning job with a new name.

    Args:
        project: The Google Cloud project ID.
        location: The region or zone where the original tuning job is located.
        tuning_job_name: The name of the original tuning job.
        new_tuning_job_name: The new name for the tuning job.

    Returns:
        The newly created HyperparameterTuningJob object.
    """

    aiplatform.init(project=project, location=location)

    tuning_job_id = "4577657336138563584"
    job = sft.SupervisedTuningJob(
        f"projects/{project}/locations/{location}/tuningJobs/{tuning_job_id}"
    )
    return job

# Example usage:
project = config.PROJECT.get('project_id')
location = config.PROJECT.get('location')
tuning_job_id = "4577657336138563584"


job = get_job_using_id(project, location, tuning_job_id)
print(job.__dict__)







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
        4. Evaluate the tuned model
        5. Plot evaluation metrics
    """
    logger.info("Starting the supervised fine-tuning pipeline.")
    try:
       


        # Step 1: Model evaluation
        evaluate_and_log_model(tuning_job)

        # Step 2: Plot evaluation metrics
        plot_and_log_metrics(tuning_job)

        logger.info("Pipeline completed successfully.")
    
    except Exception as e:
        logger.exception("An error occurred during the fine-tuning pipeline execution.")
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
