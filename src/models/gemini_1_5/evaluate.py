import logging
import yaml
import pandas as pd
from typing import List, Dict
from rouge_score import rouge_scorer
from tqdm import tqdm
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import aiplatform

def evaluate_model(tuning_job):
    """Evaluates the tuned model."""
    logging.info("Starting model evaluation.")
    try:
        with open('configs/generation_config.yaml', 'r') as file:
            gen_config = yaml.safe_load(file)
        with open('configs/dataset_config.yaml', 'r') as file:
            dataset_config = yaml.safe_load(file)

        test_dataset_path = dataset_config['test_dataset_path']
        temperature = gen_config['temperature']
        max_output_tokens = gen_config['max_output_tokens']

        # Load test data
        test_data = pd.read_csv(test_dataset_path)
        corpus = test_data.to_dict(orient='records')

        # Initialize model
        tuned_model_endpoint_name = tuning_job.tuned_model_endpoint_name
        tuned_genai_model = GenerativeModel(tuned_model_endpoint_name)

        # Evaluate model
        evaluation_df = run_evaluation(
            tuned_genai_model,
            corpus,
            temperature,
            max_output_tokens
        )

        # Compute metrics
        evaluation_stats = evaluation_df.dropna().describe()
        logging.info(f"Evaluation metrics: {evaluation_stats}")

    except Exception as e:
        logging.exception("An error occurred during model evaluation.")
        raise e

def run_evaluation(
    model: GenerativeModel,
    corpus: List[Dict],
    temperature: float,
    max_output_tokens: int
) -> pd.DataFrame:
    """Runs evaluation for the given model and data."""
    try:
        records = []
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        for item in tqdm(corpus):
            document = item.get("input_text")
            summary = item.get("output_text")
            response = model.generate_content(
                contents=document,
                generation_config=generation_config
            ).text

            scores = scorer.score(target=summary, prediction=response)

            records.append(
                {
                    "document": document,
                    "summary": summary,
                    "generated_summary": response,
                    "rouge1_fmeasure": scores["rouge1"].fmeasure,
                    "rouge2_fmeasure": scores["rouge2"].fmeasure,
                    "rougeL_fmeasure": scores["rougeL"].fmeasure,
                }
            )
        return pd.DataFrame(records)
    except Exception as e:
        logging.exception("An error occurred during evaluation.")
        raise e
