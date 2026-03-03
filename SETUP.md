# Setup Guide

## 🔧 Installation

### 1. Core Dependencies
```bash
pip install langchain langchain-community
```

### 2. Vector Stores
```bash
pip install chromadb faiss-cpu pinecone-client
```

### 3. LLM Providers
```bash
# Google Gemini
pip install langchain-google-genai

# Cohere
pip install langchain-cohere

# HuggingFace
pip install transformers torch

# Together AI
pip install together
```

### 4. Production Tools
```bash
pip install fastapi uvicorn redis pydantic
```

### 5. Utilities
```bash
pip install python-dotenv requests beautifulsoup4
```

## 🤖 Ollama Setup

### Windows
1. Download from https://ollama.ai
2. Install and run
3. Pull models:
```bash
ollama pull llama2
ollama pull mistral
ollama pull qwen3:4b
ollama pull phi
```

### Verify
```bash
ollama list
ollama run qwen3:4b "Hello"
```

## 🔑 API Keys (Optional)

### Google Gemini (Free Tier)
1. Visit https://makersuite.google.com/app/apikey
2. Create API key
3. Set environment variable:
```bash
export GOOGLE_API_KEY="your-key"
```

### Cohere (Free Trial)
1. Visit https://dashboard.cohere.com/api-keys
2. Get API key
3. Set:
```bash
export COHERE_API_KEY="your-key"
```

### HuggingFace (Free)
1. Visit https://huggingface.co/settings/tokens
2. Create token
3. Set:
```bash
export HUGGINGFACEHUB_API_TOKEN="your-token"
```

## ✅ Verify Installation

```python
# test_setup.py
from langchain_community.llms import Ollama

llm = Ollama(model="qwen3:4b")
response = llm.invoke("Say hello")
print(response)
```

Run:
```bash
python test_setup.py
```

## 📦 Requirements.txt

```
langchain==0.1.0
langchain-community==0.0.10
langchain-google-genai==0.0.5
langchain-cohere==0.0.3
chromadb==0.4.22
faiss-cpu==1.7.4
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
python-dotenv==1.0.0
redis==5.0.1
transformers==4.36.2
torch==2.1.2
```

Install all:
```bash
pip install -r requirements.txt
```

## 🐳 Docker Setup

```bash
# Pull Ollama
docker pull ollama/ollama

# Run Ollama
docker run -d -p 11434:11434 --name ollama ollama/ollama

# Pull model
docker exec ollama ollama pull qwen3:4b
```

## 🚀 Quick Test

```bash
cd langchain-complete-course
python llm-providers/01-ollama/multi_models.py
```

## 🔍 Troubleshooting

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434

# Restart
ollama serve
```

### Import errors
```bash
pip install --upgrade langchain langchain-community
```

### Model not found
```bash
ollama pull <model-name>
```

## 📚 Next Steps

1. Complete setup verification
2. Start with `llm-providers/01-ollama`
3. Progress through core concepts
4. Build real-world projects

Ready to learn! 🎓
