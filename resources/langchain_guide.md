# Langchain Comprehensive Guide

> **JavaScript / TypeScript:** If you build APIs or Next.js apps with LangChain, use the official [LangChain.js / TypeScript docs](https://docs.langchain.com/oss/javascript/langchain/overview). This guide’s examples are mostly **Python**; concepts (chains, tools, RAG) transfer across runtimes.

Complete guide to Langchain for building Generative AI applications and projects.

## Table of Contents

- [Introduction to Langchain](#introduction-to-langchain)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [LLMs and Chat Models](#llms-and-chat-models)
- [Prompts and Prompt Templates](#prompts-and-prompt-templates)
- [Chains](#chains)
- [Agents](#agents)
- [Memory](#memory)
- [Document Loaders](#document-loaders)
- [Vector Stores and Embeddings](#vector-stores-and-embeddings)
- [Retrievers](#retrievers)
- [RAG (Retrieval Augmented Generation)](#rag-retrieval-augmented-generation)
- [Real-World Projects](#real-world-projects)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to Langchain

### What is Langchain?

**Langchain** is a framework for developing applications powered by language models. It provides:
- **Modular Components**: Reusable building blocks
- **Chains**: Combine components for complex workflows
- **Agents**: LLM-powered decision-making
- **Memory**: Conversation history management
- **Vector Stores**: Document retrieval and RAG

### Why Langchain?

- **Rapid Development**: Build AI apps faster
- **Modularity**: Mix and match components
- **Integration**: Works with many LLMs and tools
- **Production Ready**: Built for real applications

### Installation

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters
# Optional: local / Hugging Face integrations
pip install langchain-huggingface sentence-transformers
```

> **API note:** Prefer `ChatOpenAI` from `langchain_openai`. The old `langchain.llms.OpenAI` completion wrapper and models like `text-davinci-003` are retired.

---

## Core Concepts

### Components

1. **LLMs**: Language models (GPT, Llama, etc.)
2. **Prompts**: Templates for LLM inputs
3. **Chains**: Sequences of operations
4. **Agents**: Autonomous decision-makers
5. **Memory**: Conversation state
6. **Vector Stores**: Document storage and retrieval

---

## Getting Started

### Basic Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Chat models are the current OpenAI path (not completion / davinci)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "Write a short article about {topic}"
)

chain = prompt | llm
result = chain.invoke({"topic": "machine learning"})
print(result.content)
```

---

## LLMs and Chat Models

### Using OpenAI

```python
from langchain_openai import ChatOpenAI

# Default: a current small chat model for tutorials
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
)

# Stronger (and costlier) option when you need it
# chat = ChatOpenAI(model="gpt-4o", temperature=0.7)
```

`temperature=0` lowers randomness. It does **not** guarantee identical outputs across providers, or that the answer is correct.

### Using Hugging Face

```python
from langchain_community.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 500}
)
```

### Using Local Models

```python
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="./models/llama-7b.gguf",
    temperature=0.7,
    n_ctx=2048
)
```

---

## Prompts and Prompt Templates

### Basic Prompt Template

```python
from langchain.prompts import PromptTemplate

template = "Tell me a {adjective} story about {topic}"
prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template
)

formatted_prompt = prompt.format(adjective="funny", topic="robots")
print(formatted_prompt)
```

### Chat Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

messages = chat_prompt.format_prompt(
    input_language="English",
    output_language="French",
    text="Hello, how are you?"
).to_messages()
```

### Few-Shot Prompting

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_template = """
Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
)

print(few_shot_prompt.format(input="big"))
```

---

## Chains

### Simple Chain

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("Python programming")
```

### Sequential Chains

```python
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate story
story_chain = LLMChain(llm=llm, prompt=story_prompt)

# Chain 2: Summarize story
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Combine
overall_chain = SimpleSequentialChain(
    chains=[story_chain, summary_chain],
    verbose=True
)

result = overall_chain.run("space exploration")
```

### Router Chains

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain

# Multiple specialized chains
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering physics questions",
        "prompt_template": "You are a physics expert. Answer: {input}"
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": "You are a math expert. Answer: {input}"
    }
]

chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=prompt_infos,
    verbose=True
)
```

---

## Agents

### What are Agents?

**Agents** use LLMs to decide which actions to take and in what order. They can use tools, search the web, run code, etc.

### Basic Agent

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for searching the internet"
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="Useful for doing math"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run
result = agent.run("What is the capital of France? Then calculate 15 * 23")
```

### Custom Tools

```python
from langchain.tools import BaseTool
from typing import Optional

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Useful for custom operations"
    
    def _run(self, query: str) -> str:
        # Tool implementation
        return f"Result for {query}"
    
    async def _arun(self, query: str) -> str:
        # Async implementation
        raise NotImplementedError

tools = [CustomTool()]
```

### Agent Types

- **ZERO_SHOT_REACT_DESCRIPTION**: General purpose, no memory
- **CONVERSATIONAL_REACT_DESCRIPTION**: With conversation memory
- **REACT_DOCSTORE**: For document question answering
- **SELF_ASK_WITH_SEARCH**: For complex reasoning

---

## Memory

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

chain.predict(input="Hi, my name is Alice")
chain.predict(input="What's my name?")
```

### Conversation Buffer Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last k exchanges
memory = ConversationBufferWindowMemory(k=2)
```

### Conversation Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

# Summarize old conversations
memory = ConversationSummaryMemory(llm=llm)
```

### Entity Memory

```python
from langchain.memory import ConversationEntityMemory

# Remember entities (people, places, etc.)
memory = ConversationEntityMemory(llm=llm)
```

---

## Document Loaders

### Loading from Files

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

# Text file
loader = TextLoader("document.txt")
documents = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# CSV
loader = CSVLoader("data.csv")
documents = loader.load()
```

### Loading from Web

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

### Loading from Directory

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./documents/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
```

---

## Vector Stores and Embeddings

### Creating Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Hugging Face embeddings (free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

### Chroma Vector Store

```python
from langchain_community.vectorstores import Chroma

# Create with persistence
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Query
docs = vectorstore.similarity_search("machine learning", k=3)
```

---

## Retrievers

### Basic Retriever

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("query")
```

### MMR Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

### Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

---

## RAG (Retrieval Augmented Generation)

### Basic RAG

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "What is machine learning?"})
print(result["result"])
```

### Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

result = qa_chain({"question": "What is RAG?"})
```

---

## Real-World Projects

### Project 1: Document Q&A System

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load documents
loader = DirectoryLoader("./documents/", glob="**/*.pdf")
documents = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
answer = qa_chain.run("What are the key findings?")
```

### Project 2: Chatbot with Memory

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chain.predict(input=user_input)
    print(f"Bot: {response}")
```

### Project 3: Code Assistant Agent

```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import PythonREPLTool

tools = [
    PythonREPLTool(),
    Tool(
        name="Search",
        func=search_function,
        description="Search for code examples"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Write Python code to sort a list")
```

---

## Best Practices

1. **Prompt Engineering**: Write clear, specific prompts
2. **Chunking**: Optimize chunk size for your use case
3. **Memory Management**: Use appropriate memory type
4. **Error Handling**: Handle LLM errors gracefully
5. **Cost Management**: Monitor token usage
6. **Caching**: Cache LLM responses when possible
7. **Testing**: Test with diverse inputs

---

## Resources

### Official Documentation

- [Langchain Documentation](https://python.langchain.com/)
- [Langchain GitHub](https://github.com/langchain-ai/langchain)
- [Langchain Cookbook](https://github.com/langchain-ai/langchain-cookbook)

### Tutorials

- [Langchain Tutorials](https://python.langchain.com/docs/get_started/introduction)
- [Langchain Examples](https://github.com/langchain-ai/langchain/tree/master/templates)

### Community

- [Langchain Discord](https://discord.gg/langchain)
- [Langchain Twitter](https://twitter.com/langchainai)

---

**Try next:** Build one chain end to end (load, split, retrieve, answer). Add agents only when a single chain is not enough.

