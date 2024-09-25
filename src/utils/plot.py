from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.metadata import context
from src.config.logging import logger
from google.cloud import aiplatform
import matplotlib.pyplot as plt
from vertexai.tuning import sft
from typing import Tuple
from typing import Dict
from typing import List 
import os


def get_metrics(tuning_job: sft.SupervisedTuningJob) -> Dict:
    """
    Fetch metrics from Tensorboard for a specific tuning job.
    
    Args:
        tuning_job (sft.SupervisedTuningJob): The tuning job for which metrics need to be retrieved.
    
    Returns:
        Dict: Dictionary containing train and eval loss metrics.
    
    Raises:
        Exception: If the metrics retrieval fails.
    """
    try:
        logger.info("Fetching metrics from Tensorboard for tuning job: %s", tuning_job.name)
        experiment_name = tuning_job.experiment.resource_name

        experiment = aiplatform.Experiment(experiment_name=experiment_name)
        filter_str = metadata_utils._make_filter_string(
            schema_title="system.ExperimentRun",
            parent_contexts=[experiment.resource_name],
        )
        experiment_run = context.Context.list(filter_str)[0]
        
        logger.info("Experiment run retrieved: %s", experiment_run.name)

        tensorboard_run_name = (
            f"{experiment.get_backing_tensorboard_resource().resource_name}/"
            f"experiments/{experiment.name}/runs/{experiment_run.name.replace(experiment.name, '')[1:]}"
        )
        tensorboard_run = aiplatform.TensorboardRun(tensorboard_run_name)
        
        logger.info("Tensorboard run retrieved: %s", tensorboard_run_name)
        metrics = tensorboard_run.read_time_series_data()

        return metrics
    except Exception as e:
        logger.exception("Failed to fetch metrics from Tensorboard.")
        raise e


def get_loss_values(metrics: Dict, metric: str = "/train_total_loss") -> Tuple[List[int], List[float]]:
    """
    Extract the steps and loss values for a specified metric from the metrics dictionary.
    
    Args:
        metrics (Dict): Dictionary containing time series data from Tensorboard.
        metric (str): The name of the metric to extract (e.g., '/train_total_loss', '/eval_total_loss').
    
    Returns:
        Tuple[List[int], List[float]]: A tuple containing two lists: steps and corresponding loss values.
    
    Raises:
        KeyError: If the specified metric is not found in the metrics.
    """
    try:
        logger.info("Extracting loss values for metric: %s", metric)
        loss_values = metrics[metric].values
        steps_loss = [loss.scalar.value for loss in loss_values]
        steps = [loss.step for loss in loss_values]
        logger.info("Loss values extracted successfully for metric: %s", metric)
        return steps, steps_loss
    except KeyError as e:
        logger.exception("The specified metric %s was not found in the metrics.", metric)
        raise e


def plot_metrics(tuning_job: sft.SupervisedTuningJob) -> None:
    """
    Plot training and evaluation loss metrics and save the plot as a PNG file.
    
    Args:
        tuning_job (sft.SupervisedTuningJob): The tuning job object containing metrics.
    
    Raises:
        Exception: If there is an error during plotting or saving the plot.
    """
    try:
        logger.info("Plotting metrics for tuning job: %s", tuning_job.name)
        
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

        # Save plot
        output_path = './data/output/gemini_1_5/loss_plot.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Plot saved successfully to %s", output_path)
    
    except Exception as e:
        logger.exception("Failed to plot metrics.")
        raise e
