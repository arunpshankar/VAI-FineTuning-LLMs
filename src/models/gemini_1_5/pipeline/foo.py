from src.models.gemini_1_5.evaluate import evaluate_model
from src.utils.plot import plot_metrics
from src.config.logging import logger
from src.config.loader import config
from google.cloud import aiplatform
from vertexai.tuning import sft
import os
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.metadata import context
from plotly.subplots import make_subplots
from src.config.logging import logger 
from src.config.loader import config
from google.cloud import aiplatform
import plotly.graph_objects as go
from vertexai.tuning import sft
import os



# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.PROJECT.get('credentials_path')

project_id = config.PROJECT.get('project_id')
location = config.PROJECT.get('location')
job = sft.SupervisedTuningJob(f"projects/{project_id}/locations/{location}/tuningJobs/4577657336138563584")
experiment_name = job.experiment.resource_name

print(experiment_name)



