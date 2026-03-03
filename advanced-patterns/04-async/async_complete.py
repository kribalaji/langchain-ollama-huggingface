import asyncio
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List

llm = Ollama(model="qwen3:4b")

# Example 1: Async LLM Calls
print("=== Async LLM Calls ===\n")

async def async_query(prompt: str) -> str:
    return await llm.ainvoke(prompt)

async def multiple_queries():
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Go?"
    ]
    
    tasks = [async_query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    for q, r in zip(queries, results):
        print(f"Q: {q}\nA: {r}\n")

asyncio.run(multiple_queries())

# Example 2: Async Chain
print("=== Async Chain ===\n")

async def async_chain_example():
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in one sentence"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    topics = ["AI", "ML", "DL"]
    tasks = [chain.ainvoke({"topic": t}) for t in topics]
    results = await asyncio.gather(*tasks)
    
    for topic, result in zip(topics, results):
        print(f"{topic}: {result['text']}")

asyncio.run(async_chain_example())

# Example 3: Async Batch Processing
print("\n=== Async Batch Processing ===\n")

async def process_batch(items: List[str], batch_size: int = 3):
    async def process_item(item: str):
        return await llm.ainvoke(f"Summarize: {item}")
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = await asyncio.gather(*[process_item(item) for item in batch])
        results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}")
    
    return results

items = [f"Item {i}" for i in range(10)]
results = asyncio.run(process_batch(items))
print(f"Processed {len(results)} items")

# Example 4: Async with Timeout
print("\n=== Async with Timeout ===\n")

async def query_with_timeout(prompt: str, timeout: int = 5):
    try:
        result = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return "Query timed out"

async def run_with_timeout():
    result = await query_with_timeout("What is async programming?", timeout=10)
    print(result)

asyncio.run(run_with_timeout())

# Example 5: Async Streaming
print("\n=== Async Streaming ===\n")

async def async_stream():
    async for chunk in llm.astream("Count from 1 to 5"):
        print(chunk, end="", flush=True)
    print()

asyncio.run(async_stream())

# Example 6: Parallel Agent Execution
print("\n=== Parallel Agents ===\n")

class AsyncAgent:
    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm
    
    async def execute(self, task: str) -> str:
        result = await self.llm.ainvoke(f"{self.name}: {task}")
        return f"{self.name} result: {result}"

async def run_parallel_agents():
    agents = [
        AsyncAgent("Researcher", llm),
        AsyncAgent("Writer", llm),
        AsyncAgent("Reviewer", llm)
    ]
    
    tasks = [agent.execute("Analyze AI trends") for agent in agents]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

asyncio.run(run_parallel_agents())
