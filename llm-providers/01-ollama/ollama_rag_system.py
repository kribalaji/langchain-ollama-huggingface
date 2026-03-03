"""RAG System with Ollama - Light & Medium Models"""
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Sample documents
docs = [
    "LangChain is a framework for developing LLM applications with chains and agents.",
    "RAG combines retrieval and generation for accurate responses using vector databases.",
    "Ollama runs models locally: llama3.2 (3B), qwen2.5 (7B), mistral (7B).",
    "Vector embeddings convert text to numerical representations for semantic search."
]

# Light model: qwen2.5:3b, Medium: mistral:7b
model = Ollama(model="qwen2.5:3b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = splitter.create_documents(docs)
vectorstore = FAISS.from_documents(splits, embeddings)

# RAG chain
template = """Use context to answer. If unknown, say so.
Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": prompt}
)

# Test
questions = [
    "What is RAG?",
    "Which Ollama models are lightweight?",
    "How do vector embeddings work?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {qa_chain.invoke(q)['result']}\n")
