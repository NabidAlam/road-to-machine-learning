# Generative AI Project Tutorial

Step-by-step tutorial: Building a RAG System for Document Q&A.

## Project: RAG System for Document Q&A

### Objective

Build a Retrieval-Augmented Generation (RAG) system that can answer questions about documents using GPT-4 and vector databases.

### Prerequisites

- Python 3.8+
- OpenAI API key
- Basic understanding of LangChain and vector databases

Copy the LangChain cells onto your machine with an API key and packages installed. They are tagged so the Study Hub accuracy suite does not call OpenAI or require PDFs.

### Step 1: Setup Environment

```python snippet-skip
# Install required packages
# pip install langchain langchain-openai langchain-community langchain-text-splitters chromadb pypdf

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Step 2: Load Documents

```python snippet-skip
# Load PDF document
loader = PyPDFLoader("document.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(f"First page: {documents[0].page_content[:200]}")
```

### Step 3: Split Documents into Chunks

```python snippet-skip
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

```python snippet-skip
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

```python snippet-skip
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
)
```

### Step 6: Create QA Chain

```python snippet-skip
# Create QA chain
# temperature=0 lowers randomness; not a correctness guarantee
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

### Step 7: Query the System

```python snippet-skip
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

```python snippet-skip
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

```python snippet-skip
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

```python snippet-skip
# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
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

```python snippet-skip
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

## Tiny local smoke (no API key)

Toy retrieve-then-answer with scikit-learn TF-IDF. Same RAG idea. No OpenAI call.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

docs = [
    "Graph neural networks pass messages along edges between nodes.",
    "Whisper is a speech recognition model that maps audio to text.",
    "Retrieval-augmented generation retrieves document chunks and feeds them to a language model.",
]
query = "retrieval-augmented generation document chunks language model"

vec = TfidfVectorizer()
X = vec.fit_transform(docs)
q = vec.transform([query])
scores = cosine_similarity(q, X).ravel()
top = int(np.argmax(scores))
context = docs[top]
# Stand-in "generation": echo the best chunk
answer = f"Based on the corpus: {context}"
print(f"top_doc={top} score={scores[top]:.3f}")
print(answer)
assert top == 2
assert scores[top] > 0
```

---

**Next**, see [Quick Reference →](generative-ai-llms-quick-reference.md) for code snippets.
