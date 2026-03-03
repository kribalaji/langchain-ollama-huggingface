from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFacePipeline
import os

# Method 1: HuggingFace Hub (requires API key)
# Get free key from https://huggingface.co/settings/tokens
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-token"

print("=== HuggingFace Hub Models ===\n")

models = [
    "google/flan-t5-base",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf"
]

for model_name in models:
    try:
        llm = HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": 0.7, "max_length": 100}
        )
        print(f"Model: {model_name}")
        response = llm.invoke("What is AI?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error with {model_name}: {e}\n")

# Method 2: Local Pipeline (no API key needed)
print("=== Local Pipeline ===\n")

try:
    from transformers import pipeline
    
    pipe = pipeline("text-generation", model="gpt2")
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    response = local_llm.invoke("The future of AI is")
    print(f"Response: {response}")
except Exception as e:
    print(f"Install transformers: pip install transformers torch")
