"""Chatbot with Memory - Ollama Models"""
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate

# Light: llama3.2:3b, Medium: qwen2.5:7b, mistral:7b
llm = Ollama(model="llama3.2:3b", temperature=0.7)

# Buffer Memory
print("=== Buffer Memory Chatbot ===\n")
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=False)

conversations = [
    "My name is Alex and I'm learning LangChain",
    "What's my name?",
    "What am I learning?"
]

for msg in conversations:
    response = chain.predict(input=msg)
    print(f"User: {msg}")
    print(f"Bot: {response}\n")

# Summary Memory (for long conversations)
print("=== Summary Memory ===\n")
summary_memory = ConversationSummaryMemory(llm=llm)
summary_chain = ConversationChain(llm=llm, memory=summary_memory)

for msg in ["I work as a data scientist", "I use Python daily", "What do you know about me?"]:
    response = summary_chain.predict(input=msg)
    print(f"User: {msg}")
    print(f"Bot: {response}\n")

print(f"Memory Summary: {summary_memory.buffer}")
