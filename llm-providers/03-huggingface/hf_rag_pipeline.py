"""HuggingFace RAG with Free Models"""
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# Set your HF token: https://huggingface.co/settings/tokens
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxx"

# Documents
docs = [
    "Python is the most popular language for AI and machine learning development.",
    "Transformers library by HuggingFace provides pre-trained models for NLP tasks.",
    "BERT, GPT, T5 are popular transformer architectures for text processing.",
    "Fine-tuning adapts pre-trained models to specific tasks with less data."
]

# Free embeddings (local, no API key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Split and vectorize
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = splitter.create_documents(docs)
vectorstore = FAISS.from_documents(splits, embeddings)

# LLM - Popular free models
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # or "mistralai/Mistral-7B-Instruct-v0.2"
    model_kwargs={"temperature": 0.3, "max_length": 200}
)

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# Test
questions = [
    "What is the most popular AI language?",
    "What does HuggingFace provide?",
    "Name some transformer models"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {qa.invoke(q)['result']}\n")
