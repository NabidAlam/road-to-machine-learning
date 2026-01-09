# Generative AI Project Tutorial

Step-by-step tutorial: Building a RAG System for Document Q&A.

## Project: RAG System for Document Q&A

### Objective

Build a Retrieval-Augmented Generation (RAG) system that can answer questions about documents using GPT-4 and vector databases.

### Prerequisites

- Python 3.8+
- OpenAI API key
- Basic understanding of LangChain and vector databases

### Step 1: Setup Environment

```python
# Install required packages
# pip install langchain openai chromadb tiktoken pypdf

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Step 2: Load Documents

```python
# Load PDF document
loader = PyPDFLoader("document.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(f"First page: {documents[0].page_content[:200]}")
```

### Step 3: Split Documents into Chunks

```python
# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### Step 4: Create Embeddings and Vector Store

```python
# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector store created")
```

### Step 5: Create Retriever

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
)
```

### Step 6: Create QA Chain

```python
# Create QA chain
llm = OpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

### Step 7: Query the System

```python
# Ask a question
query = "What is the main topic of this document?"

result = qa_chain({"query": query})

print(f"Question: {query}")
print(f"Answer: {result['result']}")
print(f"\nSources:")
for i, doc in enumerate(result['source_documents'], 1):
    print(f"{i}. {doc.page_content[:200]}...")
```

### Step 8: Improve with Better Prompting

```python
from langchain.prompts import PromptTemplate

# Create custom prompt
prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Update QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

### Step 9: Add Conversation Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Add memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Use in conversation
result = conversational_chain({"question": "What is AI?"})
print(result["answer"])

result = conversational_chain({"question": "Can you tell me more about that?"})
print(result["answer"])  # Uses previous context
```

### Step 10: Deploy with Streamlit

```python
# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Streamlit UI
st.title("Document Q&A System")
query = st.text_input("Ask a question about the document:")

if query:
    result = qa_chain({"query": query})
    st.write(result["result"])
```

### Step 11: Evaluation

```python
# Test with sample questions
test_questions = [
    "What is the main topic?",
    "Who are the key authors?",
    "What are the main conclusions?"
]

for question in test_questions:
    result = qa_chain({"query": question})
    print(f"Q: {question}")
    print(f"A: {result['result']}\n")
```

### Extensions

1. **Add Multiple Documents**: Load multiple PDFs
2. **Use Different Vector DB**: Try Pinecone or Weaviate
3. **Add Reranking**: Improve retrieval quality
4. **Add Citations**: Show source page numbers
5. **Add UI Improvements**: Better Streamlit interface

### Troubleshooting

**Issue**: Low quality answers
- **Solution**: Increase chunk overlap, adjust chunk size, improve prompts

**Issue**: Slow retrieval
- **Solution**: Use smaller embedding models, optimize vector DB

**Issue**: High costs
- **Solution**: Use GPT-3.5-turbo, cache responses, optimize prompts

---

**Next**: See [Quick Reference →](generative-ai-llms-quick-reference.md) for code snippets.
