# Setup Guide - Ollama & HuggingFace

## Ollama Setup (Windows)

### 1. Install Ollama
```bash
# Download from: https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

### 2. Pull Light Models (3B params - Fast)
```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull phi3:3b
```

### 3. Pull Medium Models (7-8B params - Better Quality)
```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
```

### 4. Pull Embedding Model
```bash
ollama pull nomic-embed-text
```

### 5. Verify Installation
```bash
ollama list
ollama run qwen2.5:3b "Hello"
```

## HuggingFace Setup

### 1. Install Dependencies
```bash
pip install langchain langchain-community
pip install huggingface-hub transformers
pip install sentence-transformers
```

### 2. Get API Token (Free)
- Visit: https://huggingface.co/settings/tokens
- Create new token (read access)
- Set environment variable:

**Windows CMD:**
```cmd
set HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxx
```

**Windows PowerShell:**
```powershell
$env:HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxx"
```

**Python:**
```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxxxxxx"
```

### 3. Popular Free Models
- `google/flan-t5-large` - Fast, instruction-tuned
- `mistralai/Mistral-7B-Instruct-v0.2` - High quality
- `HuggingFaceH4/zephyr-7b-beta` - Chat-optimized
- `sentence-transformers/all-MiniLM-L6-v2` - Embeddings (local, no API)

## Vector Stores Setup

```bash
# FAISS (lightweight, local)
pip install faiss-cpu

# Chroma (persistent, production-ready)
pip install chromadb
```

## Test Installation

```python
# Test Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="qwen2.5:3b")
print(llm.invoke("Hello"))

# Test HuggingFace
from langchain_community.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-large")
print(llm.invoke("Hello"))
```

## Model Recommendations

### For RAG Systems
- **Ollama:** qwen2.5:7b, mistral:7b
- **HuggingFace:** google/flan-t5-large
- **Embeddings:** nomic-embed-text (Ollama), all-MiniLM-L6-v2 (HF)

### For Chatbots
- **Ollama:** llama3.2:3b (fast), llama3.1:8b (quality)
- **HuggingFace:** zephyr-7b-beta, Mistral-7B-Instruct

### For Production
- **Light:** qwen2.5:3b, phi3:3b (low latency)
- **Medium:** qwen2.5:7b, mistral:7b (balanced)

## Troubleshooting

### Ollama not found
```bash
# Check if running
ollama serve

# Restart service
# Windows: Services -> Ollama -> Restart
```

### HuggingFace API errors
- Verify token is set correctly
- Check model availability
- Use local pipeline for offline work

### Memory issues
- Use smaller models (3B params)
- Reduce batch size
- Close other applications
