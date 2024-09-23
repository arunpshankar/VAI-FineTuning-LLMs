import logging
import yaml
import time
from google.cloud import aiplatform
from vertexai.preview.tuning import sft

def tune_model():
    """Tunes the Gemini model using supervised fine-tuning."""
    logging.info("Starting model tuning.")
    try:
        with open('configs/hyperparameters.yaml', 'r') as file:
            hyperparams = yaml.safe_load(file)
        with open('configs/dataset_config.yaml', 'r') as file:
            dataset_config = yaml.safe_load(file)
        with open('configs/project_config.yaml', 'r') as file:
            project_config = yaml.safe_load(file)

        tuned_model_display_name = hyperparams['tuned_model_display_name']
        epochs = hyperparams['epochs']
        learning_rate_multiplier = hyperparams['learning_rate_multiplier']

        train_dataset = dataset_config['train_dataset_path']
        validation_dataset = dataset_config['validation_dataset_path']

        # Start the tuning job
        sft_tuning_job = sft.train(
            source_model='gemini-1.5-pro-001',
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            learning_rate_multiplier=learning_rate_multiplier,
            tuned_model_display_name=tuned_model_display_name,
        )

        # Wait for the tuning job to complete
        while not sft_tuning_job.refresh().has_ended:
            logging.info("Tuning job in progress...")
            time.sleep(60)

        logging.info("Model tuning completed successfully.")
        return sft_tuning_job

    except Exception as e:
        logging.exception("An error occurred during model tuning.")
        raise e
