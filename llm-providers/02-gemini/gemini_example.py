from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Setup: Get free API key from https://makersuite.google.com/app/apikey
# os.environ["GOOGLE_API_KEY"] = "your-api-key"

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    convert_system_message_to_human=True
)

print("=== Google Gemini Example ===\n")

# Simple query
response = llm.invoke("What is machine learning?")
print(f"Response: {response.content}\n")

# Streaming
print("=== Streaming Response ===")
for chunk in llm.stream("Count from 1 to 5"):
    print(chunk.content, end="", flush=True)
print("\n")

# Batch processing
print("=== Batch Processing ===")
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Go?"
]

responses = llm.batch(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {r.content}\n")
