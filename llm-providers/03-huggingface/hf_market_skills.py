"""HuggingFace - Market-Demanded Skills"""
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxx"

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 300}
)

# Use Case 1: Code Generation
print("=== Code Generation ===\n")
code_template = """Generate Python code for: {task}
Code:"""
code_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=code_template, input_variables=["task"])
)
print(code_chain.run(task="read CSV file with pandas"))

# Use Case 2: Text Classification
print("\n=== Sentiment Analysis ===\n")
sentiment_template = """Classify sentiment as positive, negative, or neutral:
Text: {text}
Sentiment:"""
sentiment_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=sentiment_template, input_variables=["text"])
)
texts = [
    "This product is amazing!",
    "Terrible experience, very disappointed",
    "It's okay, nothing special"
]
for text in texts:
    result = sentiment_chain.run(text=text)
    print(f"{text} -> {result}")

# Use Case 3: Summarization
print("\n=== Document Summarization ===\n")
summary_template = """Summarize in one sentence:
{document}
Summary:"""
summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=summary_template, input_variables=["document"])
)
doc = """Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. It focuses on 
developing computer programs that can access data and use it to learn for themselves."""
print(summary_chain.run(document=doc))

# Use Case 4: Named Entity Recognition
print("\n=== Entity Extraction ===\n")
ner_template = """Extract person names, organizations, and locations:
Text: {text}
Entities:"""
ner_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=ner_template, input_variables=["text"])
)
text = "Elon Musk founded SpaceX in California and later moved to Texas."
print(ner_chain.run(text=text))
