from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain import hub

llm = Ollama(model="qwen3:4b")

# Define Tools
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def search(query: str) -> str:
    # Mock search
    knowledge = {
        "python": "Python is a programming language created in 1991",
        "langchain": "LangChain is a framework for LLM applications"
    }
    return knowledge.get(query.lower(), "No information found")

def get_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math. Input: math expression like '2+2'"
    ),
    Tool(
        name="Search",
        func=search,
        description="Search for information. Input: search query"
    ),
    Tool(
        name="Time",
        func=get_time,
        description="Get current time. No input needed."
    )
]

# Example 1: ReAct Agent
print("=== ReAct Agent ===\n")

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 25 * 4?"})
print(f"Result: {result['output']}\n")

# Example 2: Agent with Memory
print("=== Agent with Memory ===\n")

memory = ConversationBufferMemory(memory_key="chat_history")

agent_with_memory = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

agent_with_memory.invoke({"input": "My name is Alice"})
agent_with_memory.invoke({"input": "What's my name?"})

# Example 3: Custom Agent
print("\n=== Custom Agent ===\n")

class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    def run(self, query):
        # Decide which tool to use
        if any(op in query for op in ['+', '-', '*', '/']):
            tool_name = "Calculator"
            tool_input = query
        elif "time" in query.lower():
            tool_name = "Time"
            tool_input = ""
        elif "search" in query.lower() or "what is" in query.lower():
            tool_name = "Search"
            tool_input = query.split("what is")[-1].strip() if "what is" in query.lower() else query
        else:
            return self.llm.invoke(query)
        
        print(f"Using tool: {tool_name}")
        return self.tools[tool_name].func(tool_input)

custom_agent = SimpleAgent(llm, tools)
print(custom_agent.run("What is 10 + 5?"))
print(custom_agent.run("What is Python?"))
print(custom_agent.run("What time is it?"))

# Example 4: Multi-Step Agent
print("\n=== Multi-Step Agent ===\n")

def multi_step_task(query):
    steps = [
        ("Search", "python"),
        ("Calculator", "2023-1991"),
        ("Time", "")
    ]
    
    results = []
    for tool_name, tool_input in steps:
        tool = next(t for t in tools if t.name == tool_name)
        result = tool.func(tool_input)
        results.append(f"{tool_name}: {result}")
    
    return "\n".join(results)

print(multi_step_task("Research Python"))

# Example 5: Error Handling Agent
print("\n=== Error Handling ===\n")

class SafeAgent:
    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_retries = max_retries
    
    def run(self, query):
        for attempt in range(self.max_retries):
            try:
                # Simple routing logic
                if "calculate" in query.lower():
                    expr = query.split("calculate")[-1].strip()
                    return self.tools["Calculator"].func(expr)
                else:
                    return self.llm.invoke(query)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return "Error: Could not process request"
        return None

safe_agent = SafeAgent(llm, tools)
print(safe_agent.run("calculate 100/5"))
