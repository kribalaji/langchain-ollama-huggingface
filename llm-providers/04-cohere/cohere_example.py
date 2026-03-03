from langchain_cohere import ChatCohere
import os

# Get free trial key from https://dashboard.cohere.com/api-keys
# os.environ["COHERE_API_KEY"] = "your-api-key"

llm = ChatCohere(
    model="command",
    temperature=0.7
)

print("=== Cohere Example ===\n")

# Simple query
response = llm.invoke("Explain neural networks")
print(f"Response: {response.content}\n")

# With system message
from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful coding assistant"),
    HumanMessage(content="Write a Python function to reverse a string")
]

response = llm.invoke(messages)
print(f"Code: {response.content}\n")

# Streaming
print("=== Streaming ===")
for chunk in llm.stream("List 5 programming languages"):
    print(chunk.content, end="", flush=True)
print()
