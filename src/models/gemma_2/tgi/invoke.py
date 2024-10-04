
from src.config.logging import logger
from src.config.loader import Config 




prompt = "How would the Future of AI in 10 Years look?" 
# Overrides max_tokens and top_k parameters during inferences.
instances = [
    {
        "inputs": f"### Human: {prompt}### Assistant: ",
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    },
]
response = endpoint.predict(
    instances=instances, use_dedicated_endpoint=use_dedicated_endpoint
)

for prediction in response.predictions:
    print(prediction)