"""Ollama Models: Light vs Medium Weight Comparison"""
from langchain_community.llms import Ollama
import time

models = {
    "light": [
        ("qwen2.5:3b", "Fast, efficient, good for simple tasks"),
        ("llama3.2:3b", "Balanced speed and quality"),
        ("phi3:3b", "Microsoft's compact model")
    ],
    "medium": [
        ("qwen2.5:7b", "Better reasoning, still fast"),
        ("mistral:7b", "Strong performance, popular"),
        ("llama3.1:8b", "Meta's latest, high quality")
    ]
}

prompt = "Explain machine learning in 2 sentences"

print("=== LIGHT MODELS (3B params) ===\n")
for model_name, desc in models["light"]:
    try:
        llm = Ollama(model=model_name, temperature=0.3)
        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        print(f"Model: {model_name} - {desc}")
        print(f"Response: {response}")
        print(f"Time: {elapsed:.2f}s\n")
    except Exception as e:
        print(f"{model_name}: Not pulled. Run: ollama pull {model_name}\n")

print("=== MEDIUM MODELS (7-8B params) ===\n")
for model_name, desc in models["medium"]:
    try:
        llm = Ollama(model=model_name, temperature=0.3)
        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        print(f"Model: {model_name} - {desc}")
        print(f"Response: {response}")
        print(f"Time: {elapsed:.2f}s\n")
    except Exception as e:
        print(f"{model_name}: Not pulled. Run: ollama pull {model_name}\n")
