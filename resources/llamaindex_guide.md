# LlamaIndex Comprehensive Guide

Complete guide to LlamaIndex for building advanced generative AI projects with data indexing and retrieval.

## Table of Contents

- [Introduction to LlamaIndex](#introduction-to-llamaindex)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [Loading Data](#loading-data)
- [Indexing](#indexing)
- [Querying](#querying)
- [Retrievers](#retrievers)
- [Query Engines](#query-engines)
- [Chat Engines](#chat-engines)
- [Advanced Features](#advanced-features)
- [Real-World Projects](#real-world-projects)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to LlamaIndex

### What is LlamaIndex?

**LlamaIndex** (formerly GPT Index) is a data framework for LLM applications. It provides:
- **Data Indexing**: Efficient indexing of documents
- **Query Interface**: Natural language querying
- **Retrieval**: Smart document retrieval
- **Integration**: Works with various LLMs and data sources

### Why LlamaIndex?

- **Efficient Indexing**: Optimized for large document collections
- **Flexible Querying**: Multiple query strategies
- **Data Connectors**: Easy integration with various data sources
- **Production Ready**: Built for scalable applications

### Installation

```bash
pip install llama-index
# Or with specific integrations
pip install llama-index[openai]
pip install llama-index[all]
```

---

## Core Concepts

### Key Components

1. **Documents**: Your data (text, PDFs, etc.)
2. **Nodes**: Chunks of documents
3. **Index**: Data structure for efficient retrieval
4. **Retriever**: Finds relevant nodes
5. **Query Engine**: Answers questions using retrieved context
6. **Chat Engine**: Conversational interface

---

## Getting Started

### Basic Example

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What is machine learning?")
print(response)
```

---

## Loading Data

### From Directory

```python
from llama_index import SimpleDirectoryReader

# Load all files from directory
documents = SimpleDirectoryReader("./documents").load_data()
```

### From Files

```python
# Load specific files
documents = SimpleDirectoryReader(
    input_files=["./doc1.pdf", "./doc2.txt"]
).load_data()
```

### From URLs

```python
from llama_index import download_loader

WebPageReader = download_loader("WebPageReader")
loader = WebPageReader()
documents = loader.load_data(urls=["https://example.com/article"])
```

### From Databases

```python
from llama_index import download_loader

DatabaseReader = download_loader("DatabaseReader")
loader = DatabaseReader(uri="sqlite:///database.db")
documents = loader.load_data(query="SELECT * FROM articles")
```

### From APIs

```python
from llama_index import download_loader

NotionPageReader = download_loader("NotionPageReader")
loader = NotionPageReader(integration_token="your_token")
documents = loader.load_data(page_ids=["page-id-1", "page-id-2"])
```

---

## Indexing

### Vector Store Index

```python
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import SimpleVectorStore

# Create index
index = VectorStoreIndex.from_documents(documents)

# Save index
index.storage_context.persist(persist_dir="./storage")

# Load index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### Tree Index

```python
from llama_index import TreeIndex

# Hierarchical index
index = TreeIndex.from_documents(documents)
```

### Keyword Table Index

```python
from llama_index import KeywordTableIndex

# Keyword-based index
index = KeywordTableIndex.from_documents(documents)
```

### Composite Index

```python
from llama_index import ComposableGraph

# Combine multiple indexes
vector_index = VectorStoreIndex.from_documents(documents)
tree_index = TreeIndex.from_documents(documents)

graph = ComposableGraph.from_indices(
    [vector_index, tree_index]
)
```

---

## Querying

### Basic Query

```python
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
print(response.source_nodes)  # Source documents
```

### Streaming Response

```python
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("Explain machine learning")

for token in response.response_gen:
    print(token, end="")
```

### Custom Query

```python
from llama_index import ResponseSynthesizer
from llama_index.retrievers import VectorIndexRetriever

# Custom retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

# Custom response synthesizer
response_synthesizer = ResponseSynthesizer.from_args(
    response_mode="tree_summarize"
)

# Create query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)
```

---

## Retrievers

### Vector Retriever

```python
from llama_index.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

nodes = retriever.retrieve("query")
```

### BM25 Retriever

```python
from llama_index.retrievers import BM25Retriever

retriever = BM25Retriever.from_defaults(
    index=index,
    similarity_top_k=5
)
```

### Hybrid Retriever

```python
from llama_index.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine

# Combine vector and keyword search
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=3)

# Hybrid
from llama_index.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()
    
    def _retrieve(self, query_bundle):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        # Combine and deduplicate
        all_nodes = vector_nodes + bm25_nodes
        return list(set(all_nodes))
```

---

## Query Engines

### Basic Query Engine

```python
query_engine = index.as_query_engine()
response = query_engine.query("question")
```

### Router Query Engine

```python
from llama_index.query_engine import RouterQueryEngine
from llama_index.selectors import LLMSingleSelector

# Route to different indexes based on query
query_engine = RouterQueryEngine.from_defaults(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for semantic search"
        ),
        QueryEngineTool.from_defaults(
            query_engine=keyword_query_engine,
            description="Useful for keyword search"
        )
    ]
)
```

### Sub-Question Query Engine

```python
from llama_index.query_engine import SubQuestionQueryEngine

# Break complex query into sub-questions
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[...],
    service_context=service_context
)
```

---

## Chat Engines

### Simple Chat

```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Hello")
response = chat_engine.chat("What did I just say?")  # Remembers context
```

### Condense Question Chat

```python
from llama_index.chat_engine import CondenseQuestionChatEngine

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    verbose=True
)
```

### Context Chat Engine

```python
from llama_index.chat_engine import ContextChatEngine

chat_engine = ContextChatEngine.from_defaults(
    retriever=retriever,
    service_context=service_context
)
```

---

## Advanced Features

### Custom LLMs

```python
from llama_index.llms import OpenAI, HuggingFaceLLM

# OpenAI
llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Hugging Face
llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf"
)

# Use in service context
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm=llm)
```

### Custom Embeddings

```python
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding

# OpenAI
embeddings = OpenAIEmbedding()

# Hugging Face
embeddings = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use in service context
service_context = ServiceContext.from_defaults(embed_model=embeddings)
```

### Node Postprocessors

```python
from llama_index.postprocessor import SimilarityPostprocessor

# Filter by similarity
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor]
)
```

### Response Modes

```python
# Different response synthesis modes
query_engine = index.as_query_engine(
    response_mode="default",  # or "compact", "tree_summarize", "accumulate"
    similarity_top_k=5
)
```

---

## Real-World Projects

### Project 1: Document Q&A System

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./documents").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Interactive Q&A
while True:
    question = input("Ask a question (or 'quit'): ")
    if question.lower() == "quit":
        break
    response = query_engine.query(question)
    print(f"Answer: {response}")
    print(f"Sources: {len(response.source_nodes)} documents")
```

### Project 2: Code Documentation Assistant

```python
from llama_index import download_loader

CodeReader = download_loader("CodeReader")
loader = CodeReader()
documents = loader.load_data(file_path="./codebase/")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("How does the authentication work?")
```

### Project 3: Research Paper Assistant

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# Load research papers
documents = SimpleDirectoryReader("./papers/", recursive=True).load_data()

# Create index with metadata
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# Query with filtering
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)

response = query_engine.query("What are the key findings across all papers?")
```

---

## Best Practices

1. **Chunking**: Optimize chunk size for your documents
2. **Indexing**: Choose appropriate index type
3. **Retrieval**: Use hybrid retrieval for better results
4. **Caching**: Cache queries when possible
5. **Metadata**: Add metadata for filtering
6. **Evaluation**: Test with diverse queries

---

## Resources

### Official Documentation

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaIndex Examples](https://github.com/run-llama/llama_index/tree/main/examples)

### Tutorials

- [LlamaIndex Tutorials](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)
- [LlamaIndex Cookbook](https://github.com/run-llama/llama_index/tree/main/docs/examples)

### Community

- [LlamaIndex Discord](https://discord.gg/dGcwcsnxhU)
- [LlamaIndex Twitter](https://twitter.com/llama_index)

---

**Remember**: LlamaIndex excels at indexing and querying large document collections. Use it for document-based AI applications!

