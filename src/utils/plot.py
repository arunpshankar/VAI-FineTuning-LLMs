from google.cloud.aiplatform.metadata.context import Context
from google.cloud.aiplatform import TensorboardRun
from google.cloud.aiplatform import Experiment
from google.cloud.aiplatform import metadata
from plotly.subplots import make_subplots
from src.config.logging import logger
import plotly.graph_objects as go
from typing import Tuple
from typing import List
from typing import Dict 


def get_metrics(tuning_job, metric: str = "/train_total_loss") -> Tuple[List[int], List[float]]:
    """
    Retrieve metrics from Tensorboard for the specified tuning job and metric type.
    
    Parameters:
        tuning_job: The job object representing the tuning process.
        metric (str): The name of the metric to retrieve (default is "/train_total_loss").
    
    Returns:
        Tuple[List[int], List[float]]: A tuple containing steps and the corresponding metric values.
    """
    try:
        experiment = get_experiment_from_job(tuning_job)
        experiment_run = get_experiment_run(experiment)
        tensorboard_run_name = create_tensorboard_run_name(experiment, experiment_run)
        tensorboard_run = TensorboardRun(tensorboard_run_name)
        
        metrics = tensorboard_run.read_time_series_data()
        return extract_steps_and_metrics(metrics, metric)
    
    except Exception as e:
        logger.exception(f"Failed to retrieve metrics for metric: {metric}")
        raise e


def get_experiment_from_job(tuning_job) -> Experiment: # type: ignore
    """
    Retrieve the experiment associated with the tuning job.
    
    Parameters:
        tuning_job: The tuning job object.
    
    Returns:
        Experiment: The experiment object related to the tuning job.
    """
    try:
        return Experiment(experiment_name=tuning_job.experiment.resource_name)
    except Exception as e:
        logger.exception("Failed to retrieve experiment from tuning job.")
        raise e


def get_experiment_run(experiment) -> Context:
    """
    Retrieve the experiment run context.
    
    Parameters:
        experiment: The experiment object.
    
    Returns:
        Context: The experiment run context.
    """
    try:
        filter_str = metadata.utils._make_filter_string(
            schema_title="system.ExperimentRun",
            parent_contexts=[experiment.resource_name],
        )
        return Context.list(filter_str)[0]
    except Exception as e:
        logger.exception("Failed to retrieve experiment run.")
        raise e


def create_tensorboard_run_name(experiment, experiment_run) -> str:
    """
    Generate the Tensorboard run name for the experiment and run.
    
    Parameters:
        experiment: The experiment object.
        experiment_run: The experiment run context.
    
    Returns:
        str: The Tensorboard run name.
    """
    try:
        return f"{experiment.get_backing_tensorboard_resource().resource_name}/experiments/{experiment.name}/runs/{experiment_run.name}"
    except Exception as e:
        logger.exception("Failed to create Tensorboard run name.")
        raise e


def extract_steps_and_metrics(metrics: Dict, metric: str) -> Tuple[List[int], List[float]]:
    """
    Extract steps and metric values from the time series data.
    
    Parameters:
        metrics (Dict): The metrics time series data.
        metric (str): The metric name to extract values for.
    
    Returns:
        Tuple[List[int], List[float]]: A tuple of steps and the corresponding metric values.
    """
    try:
        loss_values = metrics[metric].values
        steps = [loss.step for loss in loss_values]
        steps_loss = [loss.scalar.value for loss in loss_values]
        return steps, steps_loss
    except Exception as e:
        logger.exception(f"Failed to extract metrics for {metric}.")
        raise e


def plot_metrics(tuning_job):
    """
    Plot the training and evaluation loss metrics for a tuning job.

    Parameters:
        tuning_job: The tuning job object to plot metrics for.
    """
    try:
        # Retrieve training and evaluation loss metrics
        train_steps, train_loss = get_metrics(tuning_job, metric="/train_total_loss")
        eval_steps, eval_loss = get_metrics(tuning_job, metric="/eval_total_loss")

        # Create subplots for train and evaluation losses
        fig = make_subplots(
            rows=1, cols=2, shared_xaxes=True, subplot_titles=("Train Loss", "Eval Loss")
        )

        # Plot training loss
        fig.add_trace(
            go.Scatter(x=train_steps, y=train_loss, name="Train Loss", mode="lines"),
            row=1, col=1,
        )

        # Plot evaluation loss
        fig.add_trace(
            go.Scatter(x=eval_steps, y=eval_loss, name="Eval Loss", mode="lines"),
            row=1, col=2,
        )

        # Update layout and display the plot
        fig.update_layout(title="Train and Eval Loss", xaxis_title="Steps", yaxis_title="Loss")
        fig.show()

    except Exception as e:
        logger.exception("Failed to plot metrics.")
        raise e
