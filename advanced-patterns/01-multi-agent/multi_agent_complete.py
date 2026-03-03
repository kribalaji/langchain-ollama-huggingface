from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List

llm = Ollama(model="qwen3:4b")

# Example 1: Simple Multi-Agent System
print("=== Multi-Agent System ===\n")

class Agent:
    def __init__(self, name: str, role: str, llm):
        self.name = name
        self.role = role
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["task"],
            template=f"You are a {role}. {{task}}"
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def execute(self, task: str) -> str:
        return self.chain.run(task=task)

# Create agents
researcher = Agent("Researcher", "research analyst", llm)
writer = Agent("Writer", "content writer", llm)
reviewer = Agent("Reviewer", "quality reviewer", llm)

# Workflow
topic = "artificial intelligence"
research = researcher.execute(f"Research key facts about {topic}")
article = writer.execute(f"Write a paragraph using: {research}")
review = reviewer.execute(f"Review this article: {article}")

print(f"Research:\n{research}\n")
print(f"Article:\n{article}\n")
print(f"Review:\n{review}\n")

# Example 2: Collaborative Agents
print("=== Collaborative Agents ===\n")

class CollaborativeSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.name: agent for agent in agents}
        self.history = []
    
    def run_workflow(self, task: str, workflow: List[tuple]):
        results = {}
        for agent_name, subtask in workflow:
            agent = self.agents[agent_name]
            # Include previous results in context
            context = f"{subtask}\nPrevious work: {results}"
            result = agent.execute(context)
            results[agent_name] = result
            self.history.append((agent_name, result))
        return results

system = CollaborativeSystem([researcher, writer, reviewer])

workflow = [
    ("Researcher", "Find 3 benefits of Python"),
    ("Writer", "Create a blog intro"),
    ("Reviewer", "Suggest improvements")
]

results = system.run_workflow("Python benefits", workflow)
for agent, output in results.items():
    print(f"{agent}:\n{output}\n")

# Example 3: Agent with Specialization
print("=== Specialized Agents ===\n")

class SpecializedAgent(Agent):
    def __init__(self, name: str, role: str, specialty: str, llm):
        super().__init__(name, role, llm)
        self.specialty = specialty
    
    def can_handle(self, task: str) -> bool:
        return self.specialty.lower() in task.lower()

code_agent = SpecializedAgent("CodeBot", "programmer", "code", llm)
data_agent = SpecializedAgent("DataBot", "data analyst", "data", llm)
doc_agent = SpecializedAgent("DocBot", "technical writer", "documentation", llm)

agents = [code_agent, data_agent, doc_agent]

tasks = [
    "Write code for a calculator",
    "Analyze data trends",
    "Create documentation for API"
]

for task in tasks:
    for agent in agents:
        if agent.can_handle(task):
            print(f"Task: {task}")
            print(f"Assigned to: {agent.name}")
            print(f"Result: {agent.execute(task)}\n")
            break

# Example 4: Agent Orchestrator
print("=== Agent Orchestrator ===\n")

class Orchestrator:
    def __init__(self, agents: List[Agent], llm):
        self.agents = agents
        self.llm = llm
    
    def route_task(self, task: str) -> Agent:
        # Simple routing logic
        if "research" in task.lower():
            return next(a for a in self.agents if a.name == "Researcher")
        elif "write" in task.lower():
            return next(a for a in self.agents if a.name == "Writer")
        else:
            return self.agents[0]
    
    def execute(self, task: str) -> Dict:
        agent = self.route_task(task)
        result = agent.execute(task)
        return {
            "task": task,
            "agent": agent.name,
            "result": result
        }

orchestrator = Orchestrator([researcher, writer, reviewer], llm)

tasks = [
    "Research machine learning",
    "Write about neural networks"
]

for task in tasks:
    output = orchestrator.execute(task)
    print(f"Task: {output['task']}")
    print(f"Agent: {output['agent']}")
    print(f"Result: {output['result']}\n")

# Example 5: Feedback Loop
print("=== Feedback Loop ===\n")

class FeedbackSystem:
    def __init__(self, creator: Agent, reviewer: Agent, max_iterations=3):
        self.creator = creator
        self.reviewer = reviewer
        self.max_iterations = max_iterations
    
    def run(self, task: str) -> str:
        content = self.creator.execute(task)
        
        for i in range(self.max_iterations):
            feedback = self.reviewer.execute(f"Review: {content}")
            
            if "approved" in feedback.lower() or "good" in feedback.lower():
                print(f"Approved after {i+1} iterations")
                return content
            
            # Improve based on feedback
            content = self.creator.execute(f"Improve based on: {feedback}")
        
        return content

feedback_system = FeedbackSystem(writer, reviewer)
final_content = feedback_system.run("Write a tagline for AI product")
print(f"Final: {final_content}")
