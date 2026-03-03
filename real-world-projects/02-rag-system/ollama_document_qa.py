"""Production RAG System - Ollama"""
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate

class DocumentQA:
    def __init__(self, model="qwen2.5:7b", persist_dir="./chroma_db"):
        self.llm = Ollama(model=model, temperature=0)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.persist_dir = persist_dir
        self.vectorstore = None
        
    def load_documents(self, path, glob_pattern="**/*.txt"):
        loader = DirectoryLoader(path, glob=glob_pattern, loader_cls=TextLoader)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(docs)
    
    def create_vectorstore(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return self.vectorstore
    
    def load_vectorstore(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def create_qa_chain(self):
        template = """Use the context to answer the question accurately.
If you don't know, say "I don't have enough information."

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def query(self, question):
        qa_chain = self.create_qa_chain()
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

# Example usage
if __name__ == "__main__":
    qa_system = DocumentQA(model="qwen2.5:7b")
    
    # Create sample documents
    sample_docs = [
        "LangChain enables building LLM applications with chains, agents, and memory.",
        "RAG systems retrieve relevant documents and generate contextual answers.",
        "Vector databases like Chroma and FAISS store embeddings for semantic search."
    ]
    
    from langchain.schema import Document
    docs = [Document(page_content=text, metadata={"source": f"doc_{i}"}) 
            for i, text in enumerate(sample_docs)]
    
    qa_system.create_vectorstore(docs)
    
    questions = [
        "What is RAG?",
        "Which vector databases are mentioned?",
        "How does LangChain help?"
    ]
    
    for q in questions:
        result = qa_system.query(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}\n")
