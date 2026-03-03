"""HuggingFace Chatbot with Streaming"""
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxx"

# Market-demand models
models = {
    "flan-t5-large": "google/flan-t5-large",  # Fast, instruction-tuned
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # High quality
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta"  # Chat-optimized
}

model_choice = "flan-t5-large"  # Change as needed

llm = HuggingFaceHub(
    repo_id=models[model_choice],
    model_kwargs={"temperature": 0.7, "max_length": 150},
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Memory keeps last 3 exchanges
memory = ConversationBufferWindowMemory(k=3)
chatbot = ConversationChain(llm=llm, memory=memory, verbose=False)

print(f"=== Chatbot with {model_choice} ===\n")

conversations = [
    "I'm building a RAG system for customer support",
    "Which vector database should I use?",
    "What was my project about?",
    "Suggest embeddings model"
]

for msg in conversations:
    print(f"\nUser: {msg}")
    print("Bot: ", end="")
    response = chatbot.predict(input=msg)
    print()  # New line after streaming
