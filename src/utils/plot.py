from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.metadata import context
from plotly.subplots import make_subplots
from src.config.logging import logger 
from src.config.loader import config
from google.cloud import aiplatform
import plotly.graph_objects as go
from vertexai.tuning import sft
import os


def get_metrics(tuning_job_id):
    """
    Get metrics from Tensorboard for a specific tuning job.

    Args:
        project_id (str): Google Cloud project ID.
        location (str): Location of the project.
        tuning_job_id (str): ID of the tuning job.

    Returns:
        dict: Dictionary containing train and eval loss metrics.
    """
    project_id = config.PROJECT.get('project_id')
    location = config.PROJECT.get('location')

    job = sft.SupervisedTuningJob(f"projects/{project_id}/locations/{location}/tuningJobs/{tuning_job_id}")
    experiment_name = job.experiment.resource_name

    experiment = aiplatform.Experiment(experiment_name=experiment_name)
    filter_str = metadata_utils._make_filter_string(
        schema_title="system.ExperimentRun",
        parent_contexts=[experiment.resource_name],
    )
    experiment_run = context.Context.list(filter_str)[0]

    tensorboard_run_name = f"{experiment.get_backing_tensorboard_resource().resource_name}/experiments/{experiment.name}/runs/{experiment_run.name.replace(experiment.name, '')[1:]}"
    tensorboard_run = aiplatform.TensorboardRun(tensorboard_run_name)
    metrics = tensorboard_run.read_time_series_data()

    def extract_metric(metric_name):
        values = metrics[metric_name].values
        return [loss.step for loss in values], [loss.scalar.value for loss in values]

    train_loss = extract_metric("/train_total_loss")
    eval_loss = extract_metric("/eval_total_loss")

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }


def plot_metrics(metrics, output_path):
    """
    Plot metrics and save the plot as a PNG file.

    Args:
        metrics (dict): Dictionary containing train and eval loss metrics.
        output_path (str): Path to save the output PNG file.
    """
    metrics = get_metrics()
    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=True, subplot_titles=("Train Loss", "Eval Loss")
    )

    fig.add_trace(
        go.Scatter(x=metrics["train_loss"][0], y=metrics["train_loss"][1], name="Train Loss", mode="lines"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=metrics["eval_loss"][0], y=metrics["eval_loss"][1], name="Eval Loss", mode="lines"),
        row=1, col=2
    )

    fig.update_layout(
        title="Train and Eval Loss",
        xaxis_title="Steps",
        yaxis_title="Loss",
        height=600,
        width=1000
    )

    fig.update_xaxes(title_text="Steps")
    fig.update_yaxes(title_text="Loss")
    output_path = './data/output/gemini_1_5/loss_plot.png'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure as a PNG file
    fig.write_image(output_path, scale=2)  # scale=2 for higher resolution
