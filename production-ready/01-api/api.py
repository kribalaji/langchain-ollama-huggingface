from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import uvicorn

app = FastAPI(title="LangChain API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = Ollama(model="qwen3:4b")

# Memory store (use Redis in production)
sessions = {}

# Models
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    response: str
    session_id: str

class ChatRequest(BaseModel):
    message: str
    session_id: str

# Endpoints
@app.get("/")
async def root():
    return {"message": "LangChain API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = llm.invoke(request.query)
        return QueryResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get or create session memory
        if request.session_id not in sessions:
            sessions[request.session_id] = ConversationBufferMemory()
        
        memory = sessions[request.session_id]
        
        # Create chain with memory
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="History: {history}\nHuman: {input}\nAI:"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        response = chain.run(input=request.message)
        
        return {
            "response": response,
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
