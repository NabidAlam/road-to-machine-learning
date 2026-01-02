# Advanced Natural Language Processing Topics

Comprehensive guide to advanced NLP techniques and architectures.

## Table of Contents

- [Advanced Transformer Architectures](#advanced-transformer-architectures)
- [Advanced Text Preprocessing](#advanced-text-preprocessing)
- [Sequence-to-Sequence Models](#sequence-to-sequence-models)
- [Advanced Embeddings](#advanced-embeddings)
- [Model Optimization](#model-optimization)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Advanced Transformer Architectures

### BERT (Bidirectional Encoder)

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors='pt')
outputs = model(**encoded)
```

### GPT (Generative Pre-trained Transformer)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Advanced Text Preprocessing

### Handling Special Cases

```python
def advanced_preprocessing(text):
    # Handle emojis
    import emoji
    text = emoji.demojize(text)
    
    # Handle mentions and hashtags
    text = re.sub(r'@\w+', '<MENTION>', text)
    text = re.sub(r'#\w+', '<HASHTAG>', text)
    
    # Handle numbers
    text = re.sub(r'\d+', '<NUMBER>', text)
    
    return text
```

---

## Sequence-to-Sequence Models

### Encoder-Decoder Architecture

```python
# Encoder
encoder_inputs = keras.Input(shape=(None,))
encoder_embedding = layers.Embedding(vocab_size, 256)(encoder_inputs)
encoder_lstm = layers.LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(None,))
decoder_embedding = layers.Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

---

## Advanced Embeddings

### Contextual Embeddings

```python
# BERT provides contextual embeddings
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Same word, different contexts
text1 = "I deposited money in the bank"
text2 = "I sat by the river bank"

# Get contextual embeddings
inputs1 = tokenizer(text1, return_tensors='pt')
inputs2 = tokenizer(text2, return_tensors='pt')

outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

# Embeddings for "bank" will be different!
```

---

## Model Optimization

### Model Quantization

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('model_name')

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Retrieval Augmented Generation (RAG)

**Retrieval Augmented Generation (RAG)** is an architecture that enhances LLM responses by retrieving relevant information from external knowledge bases before generating answers.

### What is RAG?

RAG combines:
- **Retrieval**: Finding relevant documents/information from a knowledge base
- **Augmentation**: Adding retrieved context to the prompt
- **Generation**: Using LLM to generate response based on augmented context

**Why RAG?**
- **Up-to-date Information**: Access current information not in training data
- **Domain-Specific**: Use private/custom knowledge bases
- **Reduced Hallucination**: Ground responses in retrieved facts
- **Transparency**: Can cite sources

### RAG Architecture

```
User Query
    ↓
Retrieval System (Vector Database)
    ↓
Relevant Documents Retrieved
    ↓
Augment Prompt with Context
    ↓
LLM (GPT, Llama, etc.)
    ↓
Generated Response
```

### Basic RAG Implementation

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Load and split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(your_documents)

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create LLM
llm = OpenAI(temperature=0)

# 6. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Query
query = "What is machine learning?"
result = qa_chain({"query": query})
print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.page_content[:100]}...")
```

### Advanced RAG with Langchain

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Add conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational RAG
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Multi-turn conversation
response1 = conversational_chain({"question": "What is RAG?"})
response2 = conversational_chain({"question": "How does it work?"})  # Remembers previous context
```

### RAG Components

**1. Document Loaders:**
```python
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

# Load PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Load from web
loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

**2. Text Splitting:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

**3. Embeddings:**
```python
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Or Hugging Face (free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**4. Vector Stores:**
```python
from langchain.vectorstores import FAISS, Chroma, Pinecone

# FAISS (local)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Chroma (local with persistence)
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Pinecone (cloud)
import pinecone
pinecone.init(api_key="your-key", environment="us-east-1")
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="rag-index")
```

**5. Retrievers:**
```python
# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Similarity search with score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 3}
)

# MMR (Maximum Marginal Relevance) - diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

### RAG Case Studies

**Case Study 1: Document Q&A System**

```python
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load all documents from directory
loader = DirectoryLoader("./documents/", glob="**/*.pdf")
documents = loader.load()

# Split and embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create Q&A system
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What are the key findings?"})
```

**Case Study 2: Code Documentation Assistant**

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import Language

# Load code files
loader = TextLoader("codebase.py")
documents = loader.load()

# Split by code structure
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)
chunks = python_splitter.split_documents(documents)

# Create RAG system for code Q&A
code_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="map_reduce"  # Better for long code
)
```

**Case Study 3: Research Paper Assistant**

```python
# Load research papers
loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()

# Split with metadata
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Add metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["paper_id"] = "paper_001"
    chunk.metadata["section"] = "introduction"  # Extract from structure

# Create RAG with metadata filtering
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "filter": {"paper_id": "paper_001"}}
)
```

### Optimizing Information Retrieval

**1. Better Chunking:**
```python
# Semantic chunking (preserve meaning)
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text_splitter = SemanticChunker(embeddings)
chunks = text_splitter.create_documents([text])
```

**2. Re-ranking:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compress and re-rank retrieved documents
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

**3. Hybrid Search:**
```python
# Combine keyword and semantic search
from langchain.retrievers import BM25Retriever

# Keyword search
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Semantic search
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Combine results
def hybrid_search(query):
    keyword_results = bm25_retriever.get_relevant_documents(query)
    semantic_results = semantic_retriever.get_relevant_documents(query)
    # Combine and deduplicate
    return combined_results
```

### Optimizing Generation

**1. Prompt Engineering:**
```python
from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
```

**2. Chain Types:**
```python
# "stuff" - Simple concatenation (good for small contexts)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# "map_reduce" - Process chunks separately, then combine (good for large contexts)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever
)

# "refine" - Iteratively refine answer (best quality, slower)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever
)
```

### Best Practices

1. **Chunk Size**: 500-1000 tokens (balance context vs precision)
2. **Overlap**: 10-20% overlap between chunks
3. **Embeddings**: Use domain-specific embeddings when possible
4. **Retrieval**: Use MMR for diverse results, similarity for precise
5. **Re-ranking**: Re-rank top-k results for better quality
6. **Metadata**: Store metadata for filtering and citation
7. **Evaluation**: Test with diverse queries, measure accuracy

---

## Common Pitfalls and Solutions

### Pitfall 1: Out-of-Vocabulary Words

**Solution**: Use FastText or subword tokenization (BERT)

### Pitfall 2: Long Sequences

**Solution**: Truncate or use models that handle long sequences

### Pitfall 3: Imbalanced Classes

**Solution**: Use class weights or resampling

### Pitfall 4: Poor Retrieval in RAG

**Solution**: 
- Use better embeddings
- Optimize chunk size
- Add re-ranking
- Use hybrid search

---

## Key Takeaways

1. **Transformers**: State-of-the-art for most NLP tasks
2. **Pre-trained Models**: Leverage large-scale training
3. **Contextual Embeddings**: Better than static embeddings
4. **Fine-tuning**: Adapt models to your task
5. **RAG**: Enhance LLMs with external knowledge retrieval

---

**Remember**: Transformers have revolutionized NLP - use them! RAG extends their capabilities with external knowledge.

