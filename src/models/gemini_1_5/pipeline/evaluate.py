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




