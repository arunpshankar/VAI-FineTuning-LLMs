import time  # Import the time module for tracking duration
from src.utils.common import setup_environment
from google.cloud.aiplatform import Endpoint
from src.config.logging import logger
from src.config.loader import Config
from typing import List
from typing import Dict
from typing import Any


def prepare_prompt(prompt: str, config: Config) -> List[Dict[str, Any]]:
    """
    Prepares the instances with the given prompt and generation parameters for inference.

    Args:
        prompt (str): The input prompt for the model.
        config (Config): The configuration object containing model parameters.

    Returns:
        List[Dict[str, Any]]: A list of instances with inputs and parameters for model inference.
    """
    instances = [
        {
            "inputs": f"### Human: {prompt}### Assistant: ",
            "parameters": {
                "max_new_tokens": config.GENERATION.get('max_new_tokens'),
                "temperature": config.GENERATION.get('temperature'),
                "top_p": config.GENERATION.get('top_p'),
                "top_k": config.GENERATION.get('top_k'),
            },
        }
    ]
    logger.info(f"Prompt and generation parameters prepared: {instances}")
    return instances


def make_prediction(endpoint: Endpoint, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Makes a prediction using the given endpoint and instances.

    Args:
        endpoint (Endpoint): The Vertex AI endpoint to make predictions from.
        instances (List[Dict[str, Any]]): The input instances for the model prediction.

    Returns:
        List[Dict[str, Any]]: The model's predictions.
    
    Raises:
        Exception: If the prediction fails.
    """
    try:
        logger.info("Making prediction with the model...")
        response = endpoint.predict(instances=instances, use_dedicated_endpoint=False)
        logger.info("Prediction completed successfully.")
        return response.predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def run() -> None:
    """
    Main function to load the configuration, prepare the environment, generate instances,
    and make predictions using Vertex AI.
    """
    try:
        config = Config(model_name='gemma_2', reinitialize=True)
        logger.info("Configuration loaded successfully.")
        
        # Log the generation parameters for debugging
        logger.info(f"Generation parameters: {config.GENERATION}")

        # Setup environment variables
        setup_environment(config)

        # Prepare the prompt and instances
        prompt = "How would the Future of AI in 10 Years look?"
        instances = prepare_prompt(prompt, config)

        # Create the endpoint
        endpoint = Endpoint(endpoint_name=config.GENERATION.get('endpoint_name'))
        logger.info(f"Endpoint initialized: {config.GENERATION.get('endpoint_name')}")

        # Start timing the prediction
        start_time = time.time()
        logger.info("Starting prediction...")

        # Make the prediction
        predictions = make_prediction(endpoint, instances)

        # Stop timing the prediction
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Prediction took {duration:.2f} seconds.")

        # Process and log predictions
        for prediction in predictions:
            logger.info(f"Prediction: {prediction}")

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    run()
