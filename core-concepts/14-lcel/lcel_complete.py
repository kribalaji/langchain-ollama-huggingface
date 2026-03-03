from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

llm = Ollama(model="qwen3:4b")

# Example 1: Basic LCEL Chain
print("=== Basic LCEL Chain ===\n")

prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "Python"})
print(result)

# Example 2: Multiple Steps
print("\n=== Multi-Step Chain ===\n")

prompt1 = ChatPromptTemplate.from_template("Suggest a {type} name")
prompt2 = ChatPromptTemplate.from_template("Write a tagline for {name}")

chain = (
    {"name": prompt1 | llm | StrOutputParser()}
    | prompt2
    | llm
    | StrOutputParser()
)

result = chain.invoke({"type": "tech startup"})
print(result)

# Example 3: Parallel Execution
print("\n=== Parallel Chains ===\n")

joke_chain = ChatPromptTemplate.from_template("Tell a {topic} joke") | llm
fact_chain = ChatPromptTemplate.from_template("Tell a {topic} fact") | llm

parallel_chain = RunnableParallel(
    joke=joke_chain,
    fact=fact_chain
)

result = parallel_chain.invoke({"topic": "programming"})
print(f"Joke: {result['joke']}")
print(f"Fact: {result['fact']}")

# Example 4: Passthrough
print("\n=== Passthrough ===\n")

chain = (
    RunnablePassthrough.assign(
        summary=ChatPromptTemplate.from_template("Summarize: {text}") | llm
    )
)

result = chain.invoke({"text": "LangChain is a framework for building LLM apps"})
print(f"Original: {result['text']}")
print(f"Summary: {result['summary']}")

# Example 5: Branching
print("\n=== Conditional Branching ===\n")

def route(info):
    if "code" in info["query"].lower():
        return ChatPromptTemplate.from_template("Write code for: {query}") | llm
    else:
        return ChatPromptTemplate.from_template("Explain: {query}") | llm

chain = {"query": RunnablePassthrough()} | route

print(chain.invoke("code a fibonacci function"))
print(chain.invoke("explain recursion"))
