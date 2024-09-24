import logging
from setup_environment import setup_environment
from data_preparation import prepare_data
from model_tuning import tune_model
from model_evaluation import evaluate_model
from utils.job_utils import get_job_name_with_datetime
from utils.vertex_ai_utils import initialize_vertex_ai
from utils.plotting_utils import plot_metrics

def main():
    """Main function to run the supervised fine-tuning pipeline."""
    try:
        # Set up logging
        logging.basicConfig(
            filename='logs/app.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        # Step 1: Initialize Vertex AI
        initialize_vertex_ai()

        # Step 2: Prepare Data
        prepare_data()

        # Step 3: Tune the Model
        tuning_job = tune_model()

        # Step 4: Evaluate the Model
        evaluate_model(tuning_job)

        # Step 5: Plot Metrics
        plot_metrics(tuning_job)

    except Exception as e:
        logging.exception("An error occurred in the main pipeline.")
        raise e

if __name__ == '__main__':
    main()
