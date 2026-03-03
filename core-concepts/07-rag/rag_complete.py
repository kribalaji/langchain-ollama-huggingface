from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

llm = Ollama(model="qwen3:4b")
embeddings = OllamaEmbeddings(model="qwen3:4b")

# Example 1: Basic RAG
print("=== Basic RAG System ===\n")

docs = [
    Document(page_content="LangChain is a framework for LLM applications."),
    Document(page_content="RAG combines retrieval with generation."),
    Document(page_content="Vector stores enable semantic search."),
    Document(page_content="Embeddings convert text to vectors."),
]

vectorstore = Chroma.from_documents(docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

result = qa_chain.invoke({"query": "What is RAG?"})
print(result['result'])

# Example 2: With Text Splitting
print("\n=== RAG with Text Splitting ===\n")

long_text = """
Python is a high-level programming language. It was created by Guido van Rossum in 1991.
Python emphasizes code readability. It uses significant indentation.
Python supports multiple programming paradigms including procedural, object-oriented, and functional.
The language has a comprehensive standard library.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.create_documents([long_text])
print(f"Split into {len(chunks)} chunks")

vectorstore = FAISS.from_documents(chunks, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

print(qa.invoke({"query": "Who created Python?"})['result'])

# Example 3: Conversational RAG
print("\n=== Conversational RAG ===\n")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

print(conv_chain({"question": "What is Python?"})['answer'])
print(conv_chain({"question": "When was it created?"})['answer'])

# Example 4: Custom Retrieval
print("\n=== Custom Retrieval ===\n")

retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 3, "fetch_k": 10}
)

docs = retriever.get_relevant_documents("programming paradigms")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

# Example 5: Metadata Filtering
print("\n=== Metadata Filtering ===\n")

docs_with_metadata = [
    Document(page_content="Python basics", metadata={"level": "beginner"}),
    Document(page_content="Advanced Python", metadata={"level": "advanced"}),
    Document(page_content="Python for data science", metadata={"level": "intermediate"}),
]

vectorstore = Chroma.from_documents(docs_with_metadata, embeddings)
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"level": "beginner"}, "k": 1}
)

docs = retriever.get_relevant_documents("Python")
print(f"Found: {docs[0].page_content} (Level: {docs[0].metadata['level']})")
