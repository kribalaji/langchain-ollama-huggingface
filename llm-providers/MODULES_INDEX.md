# Ollama & HuggingFace Modules - Quick Reference

## 🚀 Quick Start

### 1. Setup Ollama (Windows)
```bash
# Install: https://ollama.ai/download
ollama pull qwen2.5:3b      # Light model
ollama pull mistral:7b      # Medium model
ollama pull nomic-embed-text # Embeddings
```

### 2. Setup HuggingFace
```bash
pip install -r llm-providers/requirements.txt
# Get token: https://huggingface.co/settings/tokens
set HUGGINGFACEHUB_API_TOKEN=hf_xxx
```

---

## 📁 Module Structure

### Ollama Modules (`llm-providers/01-ollama/`)

1. **ollama_models_comparison.py** - Light vs Medium models
   - Light: qwen2.5:3b, llama3.2:3b, phi3:3b
   - Medium: qwen2.5:7b, mistral:7b, llama3.1:8b
   - Performance benchmarks

2. **ollama_chatbot_memory.py** - Chatbot with memory
   - Buffer memory for recent context
   - Summary memory for long conversations
   - Use case: Customer support, personal assistants

3. **ollama_rag_system.py** - RAG implementation
   - FAISS vector store
   - Nomic embeddings
   - Document Q&A

### HuggingFace Modules (`llm-providers/03-huggingface/`)

1. **hf_market_skills.py** - Market-demanded skills
   - Code generation
   - Sentiment analysis
   - Text summarization
   - Named entity recognition

2. **hf_chatbot_streaming.py** - Streaming chatbot
   - Models: flan-t5-large, Mistral-7B, Zephyr-7B
   - Conversation memory
   - Real-time responses

3. **hf_rag_pipeline.py** - RAG with free models
   - Local embeddings (no API key)
   - FAISS vector store
   - Production-ready

### Real-World Projects

1. **real-world-projects/01-chatbot/hf_customer_support.py**
   - Customer support chatbot
   - Conversation summarization
   - Memory management

2. **real-world-projects/02-rag-system/ollama_document_qa.py**
   - Production RAG system
   - Persistent vector store (Chroma)
   - Source tracking

---

## 🎯 Use Cases by Model

### Light Models (3B params) - Fast, Low Resource
**Best for:**
- Real-time chatbots
- High-volume API requests
- Edge deployment
- Development/testing

**Ollama Models:**
- qwen2.5:3b
- llama3.2:3b
- phi3:3b

### Medium Models (7-8B params) - Quality & Speed Balance
**Best for:**
- RAG systems
- Complex reasoning
- Production applications
- Content generation

**Ollama Models:**
- qwen2.5:7b
- mistral:7b
- llama3.1:8b

**HuggingFace Models:**
- google/flan-t5-large
- mistralai/Mistral-7B-Instruct-v0.2
- HuggingFaceH4/zephyr-7b-beta

---

## 💼 Market-Demanded Skills Covered

### ✅ RAG Systems
- `ollama_rag_system.py` - Basic RAG
- `ollama_document_qa.py` - Production RAG
- `hf_rag_pipeline.py` - HuggingFace RAG

### ✅ Chatbots
- `ollama_chatbot_memory.py` - Memory management
- `hf_chatbot_streaming.py` - Streaming responses
- `hf_customer_support.py` - Customer support

### ✅ NLP Tasks
- `hf_market_skills.py` - Classification, NER, summarization

### ✅ Vector Databases
- FAISS (lightweight, local)
- Chroma (persistent, production)

### ✅ Embeddings
- Ollama: nomic-embed-text
- HuggingFace: all-MiniLM-L6-v2 (free, local)

---

## 🏃 Running Examples

### Ollama RAG
```bash
cd llm-providers/01-ollama
python ollama_rag_system.py
```

### HuggingFace Chatbot
```bash
cd llm-providers/03-huggingface
python hf_chatbot_streaming.py
```

### Production RAG
```bash
cd real-world-projects/02-rag-system
python ollama_document_qa.py
```

---

## 📊 Model Selection Guide

| Use Case | Model | Speed | Quality | Resource |
|----------|-------|-------|---------|----------|
| Quick chatbot | qwen2.5:3b | ⚡⚡⚡ | ⭐⭐ | Low |
| Balanced chat | llama3.2:3b | ⚡⚡ | ⭐⭐⭐ | Low |
| RAG system | qwen2.5:7b | ⚡⚡ | ⭐⭐⭐⭐ | Medium |
| Production RAG | mistral:7b | ⚡ | ⭐⭐⭐⭐⭐ | Medium |
| Code generation | llama3.1:8b | ⚡ | ⭐⭐⭐⭐⭐ | Medium |

---

## 🔧 Troubleshooting

### Ollama Issues
```bash
# Check if running
ollama list

# Restart service
# Windows: Services -> Ollama -> Restart
```

### HuggingFace Issues
```python
# Test token
import os
print(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

# Use local models (no API)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
```

### Memory Issues
- Use light models (3B)
- Reduce chunk size in RAG
- Close other applications

---

## 📚 Next Steps

1. **Week 1:** Run all Ollama examples
2. **Week 2:** Implement HuggingFace RAG
3. **Week 3:** Build custom chatbot
4. **Week 4:** Deploy production RAG system

See `SETUP_GUIDE.md` for detailed installation instructions.
