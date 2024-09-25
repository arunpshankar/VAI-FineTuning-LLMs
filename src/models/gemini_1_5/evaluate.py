from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory
from src.config.logging import logger
from src.config.loader import config
from rouge_score import rouge_scorer
from random import randint
from typing import Dict 
from typing import List 
from time import sleep
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

        # Create safety settings once and pass them to the evaluation function
        safety_settings = create_safety_settings()

        # Run the evaluation process
        evaluation_df = run_evaluation(
            tuned_model,
            corpus,
            temperature,
            max_output_tokens,
            safety_settings
        )

        # Log evaluation statistics
        evaluation_stats = evaluation_df.dropna().describe()
        logger.info(f"Evaluation completed. Metrics: \n{evaluation_stats}")

        # Save evaluation results as CSV
        output_csv_path = "./data/output/gemini_1_5/evaluation_results.csv"
        evaluation_df.to_csv(output_csv_path, index=False)
        logger.info(f"Evaluation results saved to {output_csv_path}")

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
    max_output_tokens: int, 
    safety_settings: Dict[HarmCategory, HarmBlockThreshold]
) -> pd.DataFrame:
    """
    Runs evaluation on the given generative model using the test data.
    
    Parameters:
        model (GenerativeModel): The tuned generative model.
        corpus (List[Dict]): The test dataset, each containing 'input_text' and 'output_text'.
        temperature (float): The temperature setting for text generation.
        max_output_tokens (int): The maximum number of tokens for the generated content.
        safety_settings (Dict[HarmCategory, HarmBlockThreshold]): Safety settings for content generation.
    
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

            generated_summary = generate_summary(model, document, generation_config, safety_settings)
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


def create_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    """
    Creates safety settings for content generation.
    
    Returns:
        Dict[HarmCategory, HarmBlockThreshold]: Safety settings to apply.
    """
    try:
        logger.info("Creating safety settings.")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
        logger.info("Successfully created safety settings.")
        return safety_settings
    except Exception as e:
        logger.error(f"Error creating safety settings: {e}")
        raise


def generate_summary(
    model: GenerativeModel, 
    document: str, 
    generation_config: GenerationConfig, 
    safety_settings: Dict[HarmCategory, HarmBlockThreshold]
) -> str:
    """
    Generates a summary for a given document using the model.
    
    Parameters:
        model (GenerativeModel): The generative model to use for generating content.
        document (str): The input document to summarize.
        generation_config (GenerationConfig): Configuration for generation (temperature, token limit).
        safety_settings (Dict[HarmCategory, HarmBlockThreshold]): Pre-created safety settings.
    
    Returns:
        str: The generated summary.
    """
    max_retries = 3
    fallback_summary = "Summary generation failed."

    base_temperature = generation_config.to_dict().get('temperature')

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating summary for document of length {len(document)}. Attempt {attempt + 1}/{max_retries}")
            
            response = model.generate_content(
                contents=document,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            if "safety" in str(e).lower():
                logger.error(f"Safety filter triggered: {str(e)}")
            else:
                logger.error(f"Error while generating summary: {str(e)}")
            
            if attempt < max_retries - 1:
                # Increase temperature for retry
                generation_config.temperature = base_temperature + (attempt + 1) * 0.5
                
                # Exponential backoff
                sleep_time = (2 ** attempt) + (randint(0, 1000) / 1000)
                logger.info(f"Retrying in {sleep_time:.2f} seconds with temperature {generation_config.temperature}")
                sleep(sleep_time)
            else:
                logger.error("Max retries reached. Returning fallback summary.")
                return fallback_summary
    
    return fallback_summary  # Ensure that there's always a non-empty string returned
