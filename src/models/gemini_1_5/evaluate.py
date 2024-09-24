from vertexai.generative_models import GenerationConfig, GenerativeModel
from src.config.logging import logger
from src.config.loader import config
from rouge_score import rouge_scorer
from typing import List, Dict
from tqdm import tqdm
import pandas as pd


def evaluate_model(tuning_job) -> None:
    """
    Evaluates a tuned model using the test dataset and logs the evaluation metrics.
    
    Parameters:
        tuning_job: Object that contains information about the tuned model, such as the endpoint name.
    """
    logger.info("Starting model evaluation.")
    try:
        # Load configurations from config
        temperature = config.GENERATION_CONFIG.get('temperature', 0.7)
        max_output_tokens = config.GENERATION_CONFIG.get('max_output_tokens', 512)
        test_dataset_path = config.DATASET.get('test_dataset_path')

        # Load test dataset
        test_data = pd.read_csv(test_dataset_path)
        corpus = test_data.to_dict(orient='records')

        # Initialize the tuned model from the endpoint
        tuned_model = initialize_model(tuning_job.tuned_model_endpoint_name)

        # Run the evaluation process
        evaluation_df = run_evaluation(
            tuned_model,
            corpus,
            temperature,
            max_output_tokens
        )

        # Log evaluation statistics
        evaluation_stats = evaluation_df.dropna().describe()
        logger.info(f"Evaluation completed. Metrics: \n{evaluation_stats}")

    except Exception as e:
        logger.exception("An error occurred during model evaluation.")
        raise e


def initialize_model(endpoint_name: str) -> GenerativeModel:
    """
    Initializes and returns the tuned GenerativeModel.
    
    Parameters:
        endpoint_name (str): The name of the tuned model's endpoint.
    
    Returns:
        GenerativeModel: Initialized generative model object.
    """
    try:
        logger.info(f"Initializing model at endpoint: {endpoint_name}")
        return GenerativeModel(endpoint_name)
    except Exception as e:
        logger.exception(f"Failed to initialize model at endpoint: {endpoint_name}")
        raise e


def run_evaluation(
    model: GenerativeModel,
    corpus: List[Dict],
    temperature: float,
    max_output_tokens: int
) -> pd.DataFrame:
    """
    Runs evaluation on the given generative model using the test data.
    
    Parameters:
        model (GenerativeModel): The tuned generative model.
        corpus (List[Dict]): The test dataset, each containing 'input_text' and 'output_text'.
        temperature (float): The temperature setting for text generation.
        max_output_tokens (int): The maximum number of tokens for the generated content.
    
    Returns:
        pd.DataFrame: DataFrame containing evaluation results including ROUGE scores.
    """
    try:
        records = []
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        logger.info(f"Running evaluation on {len(corpus)} examples.")
        
        for item in tqdm(corpus):
            document = item.get("input_text")
            summary = item.get("output_text")

            generated_summary = generate_summary(model, document, generation_config)
            scores = scorer.score(target=summary, prediction=generated_summary)

            records.append(
                {
                    "document": document,
                    "summary": summary,
                    "generated_summary": generated_summary,
                    "rouge1_fmeasure": scores["rouge1"].fmeasure,
                    "rouge2_fmeasure": scores["rouge2"].fmeasure,
                    "rougeL_fmeasure": scores["rougeL"].fmeasure,
                }
            )

        return pd.DataFrame(records)

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        raise e


def generate_summary(model: GenerativeModel, document: str, generation_config: GenerationConfig) -> str:
    """
    Generates a summary for a given document using the model.
    
    Parameters:
        model (GenerativeModel): The generative model to use for generating content.
        document (str): The input document to summarize.
        generation_config (GenerationConfig): Configuration for generation (temperature, token limit).
    
    Returns:
        str: The generated summary.
    """
    try:
        logger.info(f"Generating summary for document of length {len(document)}.")
        response = model.generate_content(
            contents=document,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        logger.exception(f"Error while generating summary for document: {document[:100]}...")
        raise e
