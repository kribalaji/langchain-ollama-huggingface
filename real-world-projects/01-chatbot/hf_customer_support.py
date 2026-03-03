"""Customer Support Chatbot - HuggingFace"""
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxx"

class SupportChatbot:
    def __init__(self, model="google/flan-t5-large"):
        self.llm = HuggingFaceHub(
            repo_id=model,
            model_kwargs={"temperature": 0.6, "max_length": 200}
        )
        
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=150
        )
        
        template = """You are a helpful customer support assistant.
Be polite, concise, and solution-oriented.

Current conversation:
{history}

Customer: {input}
Assistant:"""
        
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
    
    def chat(self, message):
        return self.chain.predict(input=message)
    
    def get_conversation_summary(self):
        return self.memory.moving_summary_buffer

# Example usage
if __name__ == "__main__":
    bot = SupportChatbot()
    
    print("=== Customer Support Chatbot ===\n")
    
    conversations = [
        "Hi, I can't log into my account",
        "I tried resetting password but didn't receive email",
        "My email is user@example.com",
        "Yes, I checked spam folder",
        "What should I do now?"
    ]
    
    for msg in conversations:
        print(f"Customer: {msg}")
        response = bot.chat(msg)
        print(f"Assistant: {response}\n")
    
    print(f"Conversation Summary: {bot.get_conversation_summary()}")
