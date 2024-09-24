


from src.models.gemini_1_5.evaluate import evaluate_model
from src.models.gemini_1_5.prep import prepare_data

from src.models.gemini_1_5.tune import tune_model
from src.config.logging import logger 




from utils.job_utils import get_job_name_with_datetime
from utils.vertex_ai_utils import initialize_vertex_ai
from utils.plotting_utils import plot_metrics


def run():
    """Main function to run the supervised fine-tuning pipeline."""
    try:
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
        logger.exception("An error occurred in the main pipeline.")
        raise e

if __name__ == '__main__':
    run()
