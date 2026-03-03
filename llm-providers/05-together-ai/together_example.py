from langchain_together import Together
import os

# Get API key from https://api.together.xyz/
# os.environ["TOGETHER_API_KEY"] = "your-api-key"

# Multiple models available
models = {
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "codellama": "codellama/CodeLlama-34b-Instruct-hf"
}

print("=== Together AI - Multiple Models ===\n")

for name, model_id in models.items():
    try:
        llm = Together(
            model=model_id,
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"[{name.upper()}]")
        response = llm.invoke("What is recursion?")
        print(f"{response}\n")
    except Exception as e:
        print(f"Error with {name}: {e}\n")

# Streaming example
print("=== Streaming ===")
llm = Together(model=models["mistral-7b"])
for chunk in llm.stream("Explain async programming"):
    print(chunk, end="", flush=True)
print()
