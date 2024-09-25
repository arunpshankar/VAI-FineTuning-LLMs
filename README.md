# Vertex AI LLM Fine-Tuning Examples

This repository provides clean and well-documented example code for fine-tuning large language models (LLMs) supported by Vertex AI. It includes various fine-tuning paradigms for:

- **Google's Proprietary Foundational Models**: Examples include **Gemini 1.5 Pro**.
- **Supported Open-Source Models**: Accessible through **Model Garden**, such as **Llama 3.1**, **Gemma 2**, and others.
- **Supported Third-Party Models**: Available via **Model Garden** with comprehensive fine-tuning examples.
- **Additional Models**: Examples of other models supported for fine-tuning within the Vertex AI ecosystem.

This repository is designed to demonstrate best practices for LLM fine-tuning with detailed explanations and optimized workflows.

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