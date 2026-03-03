from langchain_community.llms import Ollama

# Multiple Ollama models
models = {
    "llama2": Ollama(model="llama2"),
    "mistral": Ollama(model="mistral"),
    "qwen": Ollama(model="qwen3:4b"),
    "phi": Ollama(model="phi")
}

prompt = "Explain LangChain in one sentence"

print("=== Testing Multiple Ollama Models ===\n")

for name, llm in models.items():
    try:
        print(f"[{name.upper()}]")
        response = llm.invoke(prompt)
        print(f"{response}\n")
    except Exception as e:
        print(f"Error with {name}: {e}\n")

# Temperature comparison
print("=== Temperature Effect ===\n")
for temp in [0.0, 0.5, 1.0]:
    llm = Ollama(model="qwen3:4b", temperature=temp)
    print(f"Temperature {temp}:")
    print(llm.invoke("Write a creative tagline for AI"))
    print()
