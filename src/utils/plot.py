from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.metadata import context
from plotly.subplots import make_subplots
from src.config.logging import logger 
from src.config.loader import config
from google.cloud import aiplatform
import plotly.graph_objects as go
from vertexai.tuning import sft
import matplotlib.pyplot as plt
import os


def get_metrics(tuning_job):
    """
    Get metrics from Tensorboard for a specific tuning job.

    Args:
        project_id (str): Google Cloud project ID.
        location (str): Location of the project.
        tuning_job_id (str): ID of the tuning job.

    Returns:
        dict: Dictionary containing train and eval loss metrics.
    """

    experiment_name = tuning_job.experiment.resource_name

    experiment = aiplatform.Experiment(experiment_name=experiment_name)
    filter_str = metadata_utils._make_filter_string(
        schema_title="system.ExperimentRun",
        parent_contexts=[experiment.resource_name],
    )
    experiment_run = context.Context.list(filter_str)[0]
    print(experiment_run)

    tensorboard_run_name = f"{experiment.get_backing_tensorboard_resource().resource_name}/experiments/{experiment.name}/runs/{experiment_run.name.replace(experiment.name, '')[1:]}"
    tensorboard_run = aiplatform.TensorboardRun(tensorboard_run_name)
    print(tensorboard_run, '<<<')
    metrics = tensorboard_run.read_time_series_data()

    return metrics


def get_loss_values(metrics, metric: str = "/train_total_loss"):
    """
    Get metrics from Tensorboard.

    Args:
      metric: metric name, eg. /train_total_loss or /eval_total_loss.
    Returns:
      steps: list of steps.
      steps_loss: list of loss values.
    """
    loss_values = metrics[metric].values
    steps_loss = []
    steps = []
    for loss in loss_values:
        steps_loss.append(loss.scalar.value)
        steps.append(loss.step)
    return steps, steps_loss

def plot_metrics(tuning_job):
    """
    Plot metrics using Matplotlib and save the plot as a PNG file.
    
    Args:
        tuning_job: The tuning job object containing metrics.
    """
    metrics = get_metrics(tuning_job)
    train_steps, train_loss = get_loss_values(metrics, metric="/train_total_loss")
    eval_steps, eval_loss = get_loss_values(metrics, metric="/eval_total_loss")
    
    plt.figure(figsize=(12, 6))
    
    # Train Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    # Eval Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_loss, label='Eval Loss', color='orange')
    plt.title('Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle('Train and Eval Loss', fontsize=16)
    plt.tight_layout()
    
    output_path = './data/output/gemini_1_5/loss_plot.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")
