from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import json
from datetime import datetime

class ProductionChatbot:
    def __init__(self, model="qwen3:4b", personality="helpful assistant"):
        self.llm = Ollama(model=model, temperature=0.7)
        self.memory = ConversationBufferWindowMemory(k=5)
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=f"You are a {personality}.\n\nHistory: {{history}}\n\nHuman: {{input}}\nAI:"
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt
        )
        
        self.chat_history = []
    
    def chat(self, message: str) -> str:
        response = self.conversation.predict(input=message)
        
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": message,
            "bot": response
        })
        
        return response
    
    def save_history(self, filename="chat_history.json"):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
    
    def clear_memory(self):
        self.memory.clear()
        self.chat_history = []

if __name__ == "__main__":
    bot = ProductionChatbot(personality="friendly tech expert")
    
    print("Chatbot ready! Commands: 'quit', 'clear', 'save'\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            bot.save_history()
            break
        elif user_input.lower() == 'clear':
            bot.clear_memory()
            print("Memory cleared!\n")
            continue
        elif user_input.lower() == 'save':
            bot.save_history()
            print("History saved!\n")
            continue
        
        response = bot.chat(user_input)
        print(f"Bot: {response}\n")
