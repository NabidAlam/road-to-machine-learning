# Generative AI & LLMs Advanced Topics

Advanced topics in Generative AI and Large Language Model applications for production and research.

## Table of Contents

- [Advanced Prompt Engineering](#advanced-prompt-engineering)
- [Advanced RAG Techniques](#advanced-rag-techniques)
- [Agent Architectures](#agent-architectures)
- [Model Optimization for Production](#model-optimization-for-production)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Cost Optimization Strategies](#cost-optimization-strategies)
- [Security and Safety](#security-and-safety)
- [Scaling GenAI Applications](#scaling-genai-applications)
- [Resources and Further Reading](#resources-and-further-reading)

---

## Advanced Prompt Engineering

### Prompt Chaining

**Concept**: Break complex tasks into a series of connected prompts.

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Chain 1: Extract key points
extract_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract 5 key points from: {text}"
)

# Chain 2: Summarize
summarize_prompt = PromptTemplate(
    input_variables=["key_points"],
    template="Summarize these key points: {key_points}"
)

# Create chains
extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Execute chain
key_points = extract_chain.run(text="Long article...")
summary = summarize_chain.run(key_points=key_points)
```

### Self-Consistency

**Concept**: Generate multiple responses and select the most consistent one.

```python
def self_consistency_prompting(prompt, n=5):
    """Generate multiple responses and find consensus"""
    responses = []
    for _ in range(n):
        response = llm.generate(prompt, temperature=0.7)
        responses.append(response)
    
    # Find most common answer
    from collections import Counter
    most_common = Counter(responses).most_common(1)[0][0]
    return most_common
```

### Tree of Thoughts

**Concept**: Explore multiple reasoning paths and select the best.

```python
def tree_of_thoughts(problem, depth=3):
    """Explore multiple reasoning paths"""
    # Generate initial thoughts
    thoughts = generate_thoughts(problem)
    
    for level in range(depth):
        # Evaluate each thought
        evaluated = evaluate_thoughts(thoughts)
        # Expand best thoughts
        thoughts = expand_thoughts(evaluated[:3])
    
    return best_thought(thoughts)
```

---

## Advanced RAG Techniques

### Hybrid Search

**Concept**: Combine semantic search with keyword search.

```python
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS

# Semantic retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Keyword retriever
bm25_retriever = BM25Retriever.from_texts(texts)

# Hybrid retrieval
def hybrid_search(query, alpha=0.5):
    semantic_docs = vector_retriever.get_relevant_documents(query)
    keyword_docs = bm25_retriever.get_relevant_documents(query)
    
    # Combine with weighted scores
    combined = combine_results(semantic_docs, keyword_docs, alpha)
    return combined
```

### Query Rewriting

**Concept**: Improve retrieval by rewriting queries.

```python
def rewrite_query(original_query):
    """Rewrite query for better retrieval"""
    rewrite_prompt = f"""
    Rewrite this query to be more specific and searchable:
    Original: {original_query}
    Rewritten:
    """
    rewritten = llm(rewrite_prompt)
    return rewritten

# Use rewritten query for retrieval
rewritten = rewrite_query("Tell me about AI")
docs = retriever.get_relevant_documents(rewritten)
```

### Reranking

**Concept**: Improve retrieval quality by reranking results.

```python
from sentence_transformers import CrossEncoder

# Load reranker model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, documents, top_k=3):
    """Rerank documents using cross-encoder"""
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by scores
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

### Multi-Query Retrieval

**Concept**: Generate multiple query variations for better coverage.

```python
def multi_query_retrieval(original_query):
    """Generate multiple query variations"""
    query_prompt = f"""
    Generate 3 different ways to search for this information:
    {original_query}
    """
    variations = llm(query_prompt)
    
    # Retrieve for each variation
    all_docs = []
    for variation in variations:
        docs = retriever.get_relevant_documents(variation)
        all_docs.extend(docs)
    
    # Deduplicate and rerank
    unique_docs = deduplicate(all_docs)
    return rerank_documents(original_query, unique_docs)
```

---

## Agent Architectures

### ReAct with Memory

**Concept**: Combine reasoning, acting, and memory.

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create agent with memory
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
```

### Hierarchical Agents

**Concept**: Agents with different levels of abstraction.

```python
# High-level planner agent
planner = Agent(
    role="Planner",
    goal="Break down complex tasks into subtasks"
)

# Mid-level coordinator agent
coordinator = Agent(
    role="Coordinator",
    goal="Coordinate execution of subtasks"
)

# Low-level executor agents
executors = [
    Agent(role="Researcher", goal="Research information"),
    Agent(role="Writer", goal="Write content"),
    Agent(role="Reviewer", goal="Review and improve")
]
```

### Meta-Agents

**Concept**: Agents that manage other agents.

```python
class MetaAgent:
    def __init__(self):
        self.sub_agents = []
        self.task_queue = []
    
    def delegate(self, task):
        """Delegate task to appropriate sub-agent"""
        best_agent = self.select_agent(task)
        return best_agent.execute(task)
    
    def select_agent(self, task):
        """Select best agent for task"""
        # Use LLM to match task to agent capabilities
        return self.llm.select_agent(task, self.sub_agents)
```

---

## Model Optimization for Production

### Caching

**Concept**: Cache LLM responses to reduce costs and latency.

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())

# Responses with same prompt are cached
response1 = llm("What is AI?")  # Calls API
response2 = llm("What is AI?")  # Returns cached result
```

### Streaming

**Concept**: Stream responses for better UX.

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Stream responses
llm = OpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

response = llm("Write a long story...")  # Streams as it generates
```

### Batch Processing

**Concept**: Process multiple requests efficiently.

```python
def batch_process(prompts, batch_size=10):
    """Process prompts in batches"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = llm.batch(batch)
        results.extend(batch_results)
    return results
```

---

## Evaluation and Benchmarking

### RAG Evaluation

```python
from ragas import evaluate
from datasets import Dataset

# Prepare data
dataset = Dataset.from_dict({
    "question": ["What is AI?"],
    "contexts": [["AI is...", "Machine learning..."]],
    "answer": ["AI is artificial intelligence"],
    "ground_truth": ["AI is artificial intelligence"]
})

# Evaluate
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

### Agent Evaluation

```python
def evaluate_agent(agent, test_cases):
    """Evaluate agent on test cases"""
    results = []
    for case in test_cases:
        response = agent.run(case["input"])
        score = evaluate_response(response, case["expected"])
        results.append(score)
    return np.mean(results)
```

---

## Cost Optimization Strategies

### Model Selection

```python
# Use smaller models when possible
models = {
    "simple": "gpt-3.5-turbo",  # Cheaper
    "complex": "gpt-4"  # More expensive
}

def select_model(task_complexity):
    if task_complexity < 0.5:
        return models["simple"]
    return models["complex"]
```

### Prompt Optimization

```python
# Shorter prompts = lower costs
# Use system messages efficiently
# Cache common prompts
```

### Token Management

```python
def optimize_tokens(text, max_tokens=1000):
    """Reduce token count while preserving meaning"""
    # Summarize if too long
    if count_tokens(text) > max_tokens:
        return summarize(text, max_tokens)
    return text
```

---

## Security and Safety

### Input Validation

```python
def validate_input(user_input):
    """Validate and sanitize user input"""
    # Check length
    if len(user_input) > 10000:
        raise ValueError("Input too long")
    
    # Check for injection attempts
    if contains_injection(user_input):
        raise ValueError("Invalid input")
    
    return sanitize(user_input)
```

### Output Filtering

```python
def filter_output(response):
    """Filter harmful or inappropriate content"""
    # Check for toxicity
    if is_toxic(response):
        return "I cannot provide that response."
    
    # Check for PII
    if contains_pii(response):
        return remove_pii(response)
    
    return response
```

---

## Scaling GenAI Applications

### Load Balancing

```python
from langchain.llms import OpenAI

# Multiple API keys for load distribution
llms = [
    OpenAI(openai_api_key=key1),
    OpenAI(openai_api_key=key2),
    OpenAI(openai_api_key=key3)
]

def get_llm():
    """Round-robin load balancing"""
    return llms[request_count % len(llms)]
```

### Async Processing

```python
import asyncio
from langchain.llms import OpenAI

async def async_generate(prompts):
    """Process prompts asynchronously"""
    llm = OpenAI()
    tasks = [llm.agenerate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

---

## Resources and Further Reading

- [Advanced RAG Techniques](https://python.langchain.com/docs/use_cases/question_answering/)
- [Agent Architectures](https://python.langchain.com/docs/modules/agents/)
- [Production Best Practices](https://python.langchain.com/docs/production/)
- [Evaluation Methods](https://python.langchain.com/docs/guides/evaluation/)

---

**Next**: See [Project Tutorial →](generative-ai-llms-project-tutorial.md) for hands-on implementation.
