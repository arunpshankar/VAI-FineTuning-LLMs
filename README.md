# Fine-Tuning LLMs Using Vertex AI

This repository encapsulates code for fine-tuning large language models (LLMs) supported within Vertex AI. It covers various fine-tuning techniques for:

- **Foundational models** like **Gemini 1.5 Pro**
- **Open-source models** available through **Model Garden**
- **Third-party models** accessible via Model Garden
- Additional models supported for fine-tuning

## Setup Instructions

### Clone the repository:
```bash
git clone https://github.com/arunpshankar/VAI-FineTuning-LLMs.git
cd VAI-FineTuning-LLMs
```

### Create a virtual environment:
```bash
python -m venv .vai-finetuning-llms
source .vai-finetuning-llms/bin/activate
```

### Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Set environment variables:
```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PYTHONPATH:.
```

### Service credentials setup:
1. Create a folder named `credentials` in the root of the repo:
   ```bash
   mkdir credentials
   ```
2. Download your GCP service credentials and save the JSON file as `key.json` inside the `credentials` folder.

### Configurations:
- The `configs` folder contains all necessary YAML files for setting up fine-tuning.
- Currently, we support **Gemini 1.5 Pro** for fine-tuning.

## Running the Pipelines

To initiate fine-tuning and evaluation, use the following commands:

### Fine-tuning:
```bash
python src/models/gemini_1_5/pipeline/tuning_pipeline.py
```

### Evaluation:
```bash
python src/models/gemini_1_5/pipeline/evaluation_pipeline.py
```

### Example: Gemini 1.5 Pro Fine-Tuning
The included example demonstrates how to fine-tune **Gemini 1.5 Pro** for **abstract summarization**. The input consists of a context and a prompt, while the output is the expected summary or completion of the provided text. This example utilizes **LoRA** (Low-Rank Adaptation) for fine-tuning.