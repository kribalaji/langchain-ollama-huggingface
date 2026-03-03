from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class SQLQuery(BaseModel):
    query: str = Field(description="The SQL query")
    explanation: str = Field(description="Explanation of the query")
    tables: List[str] = Field(description="Tables used in the query")

class SQLGenerator:
    def __init__(self, model="qwen3:4b"):
        self.llm = Ollama(model=model, temperature=0.1)
        
        self.schema = """
        Tables:
        - users (id, name, email, created_at)
        - orders (id, user_id, product, amount, order_date)
        - products (id, name, price, category)
        """
    
    def generate_query(self, natural_language: str) -> str:
        prompt = PromptTemplate(
            input_variables=["schema", "query"],
            template="""Given this database schema:
{schema}

Generate a SQL query for: {query}

Return only the SQL query, nothing else."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(schema=self.schema, query=natural_language)
        return result.strip()
    
    def explain_query(self, sql_query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Explain this SQL query in simple terms:\n{query}"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(query=sql_query)
    
    def optimize_query(self, sql_query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Optimize this SQL query:\n{query}\n\nOptimized version:"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(query=sql_query)

if __name__ == "__main__":
    generator = SQLGenerator()
    
    queries = [
        "Get all users who made orders in the last 30 days",
        "Find the top 5 products by total sales",
        "Count orders per user"
    ]
    
    for nl_query in queries:
        print(f"\nNatural Language: {nl_query}")
        sql = generator.generate_query(nl_query)
        print(f"SQL: {sql}")
        explanation = generator.explain_query(sql)
        print(f"Explanation: {explanation}")
        print("-" * 50)
