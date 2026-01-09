# Generative AI & LLMs Quick Reference

Quick reference for Generative AI, LLMs, RAG, and AI agents.

## Table of Contents

- [Prompt Engineering](#prompt-engineering)
- [Vector Databases](#vector-databases)
- [RAG Systems](#rag-systems)
- [LangChain](#langchain)
- [AI Agents](#ai-agents)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)

---

## Prompt Engineering

### Basic Prompt

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning"}
    ],
    temperature=0.7,
    max_tokens=500
)
```

### Few-Shot Prompting

```python
prompt = """
Example 1:
Input: "I love this product"
Sentiment: Positive

Example 2:
Input: "This is terrible"
Sentiment: Negative

Input: "It's okay"
Sentiment:
"""
```

### Chain-of-Thought

```python
prompt = """
Solve this step by step:
Question: If a train travels 120 km in 2 hours, what's its speed?

Let me think step by step:
1. Distance = 120 km
2. Time = 2 hours
3. Speed = Distance / Time
4. Speed = 120 / 2 = 60 km/h
"""
```

### Generative Configuration

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    temperature=0.7,      # 0.0-2.0, controls randomness
    top_p=0.9,           # 0.0-1.0, nucleus sampling
    top_k=50,            # Integer, top-k sampling
    frequency_penalty=0.5,  # -2.0 to 2.0, reduce repetition
    presence_penalty=0.3,   # -2.0 to 2.0, encourage new topics
    max_tokens=500
)
```

---

## Vector Databases

### ChromaDB

```python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=["Document 1", "Document 2"],
    ids=["id1", "id2"],
    embeddings=[[0.1, 0.2], [0.3, 0.4]]
)

# Query
results = collection.query(
    query_embeddings=[[0.15, 0.25]],
    n_results=2
)
```

### Pinecone

```python
import pinecone
from langchain.vectorstores import Pinecone

# Initialize
pinecone.init(api_key="your-api-key", environment="us-east-1")

# Create index
pinecone.create_index("documents", dimension=1536)

# Create vector store
vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="documents"
)

# Query
results = vectorstore.similarity_search("query", k=3)
```

### FAISS

```python
from langchain.vectorstores import FAISS

# Create vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Query
results = vectorstore.similarity_search("query", k=3)
```

---

## RAG Systems

### Basic RAG

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

# Query
result = qa_chain({"query": "What is AI?"})
```

### RAG with Sources

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "What is AI?"})
print(result["result"])
print(result["source_documents"])
```

### Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    retriever=retriever,
    memory=memory
)

result = chain({"question": "What is AI?"})
```

---

## LangChain

### Basic LLM

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
response = llm("Explain machine learning")
```

### Prompt Templates

```python
from langchain.prompts import PromptTemplate

template = "Tell me about {topic}"
prompt = PromptTemplate(input_variables=["topic"], template=template)
response = llm(prompt.format(topic="AI"))
```

### Chains

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(topic="AI")
```

### Document Loaders

```python
from langchain.document_loaders import PyPDFLoader, TextLoader

# PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Text
loader = TextLoader("document.txt")
documents = loader.load()
```

### Text Splitters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

---

## AI Agents

### Basic Agent

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Search the web"
    )
]

agent = initialize_agent(
    tools,
    OpenAI(),
    agent="zero-shot-react-description",
    verbose=True
)

result = agent.run("What is the weather today?")
```

### ReAct Agent

```python
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "What is AI?"})
```

### LangGraph Agent

```python
from langgraph.graph import StateGraph, END

# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]

# Create graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()
```

---

## Common Patterns

### Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### Streaming

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = OpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

### Error Handling

```python
from langchain.schema import OutputParserException

try:
    result = chain.run(input)
except OutputParserException as e:
    print(f"Error: {e}")
```

---

## Best Practices

### Prompt Engineering
- Be specific and clear
- Provide context
- Use examples (few-shot)
- Specify output format
- Iterate and refine

### RAG Systems
- Chunk size: 500-1000 tokens
- Chunk overlap: 10-20%
- Retrieve 3-5 documents
- Rerank for better quality
- Add citations

### Cost Optimization
- Use GPT-3.5-turbo when possible
- Cache common queries
- Optimize prompt length
- Batch requests
- Monitor usage

### Security
- Validate inputs
- Filter outputs
- Rate limiting
- API key security
- Data privacy

---

**See Also:**
- [Complete Guide →](generative-ai-llms.md)
- [Advanced Topics →](generative-ai-llms-advanced-topics.md)
- [Project Tutorial →](generative-ai-llms-project-tutorial.md)
