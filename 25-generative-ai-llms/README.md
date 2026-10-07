# Module 25: Generative AI & Modern LLM Applications

Build LLM demos with prompt engineering, vector search, RAG, and agents only after eval. Do Days 1–6 (prompts, retrieval, faithfulness checks, guardrails) before tool-using agent loops. Treat demos as demos until you add retrieval eval, faithfulness checks, and monitoring. Prefer staging language before any live side-effect tools.

**How this module fits:** Stage 7 in the main README. Best after Module 12 (NLP) or alongside it if you learn by building. **This folder is a hub:** [generative-ai-llms.md](generative-ai-llms.md) holds the Day 1-7 spine. Depth lives in [resources/generative_ai_comprehensive_guide.md](../resources/generative_ai_comprehensive_guide.md), [resources/rag_comprehensive_guide.md](../resources/rag_comprehensive_guide.md), and the guides below.

## Day 1-7 study spine (ordered)

Work these in order. Details and depth links are in [generative-ai-llms.md](generative-ai-llms.md#day-1-7-study-spine).

| Day | Focus |
|-----|--------|
| 1 | Prompts + structured outputs (schema over “please JSON”) |
| 2 | Embeddings + vector store basics |
| 3 | Hybrid RAG (BM25 + vector + rerank pointer) + faithfulness / retrieval eval (not BLEU) |
| 4 | Context engineering + prompt caching / cost |
| 5 | PEFT decision: Prompt, then RAG, then LoRA / QLoRA ([NLP PEFT](../12-natural-language-processing/nlp-advanced-topics.md#parameter-efficient-fine-tuning-peft), [fine-tuning guide](../resources/transformer_fine_tuning_guide.md#peft-decision-path-prompt-then-rag-then-loraqlora)) |
| 6 | Guardrails + prompt injection before side-effect tools. MCP overview + permissions ([agents guide](../resources/ai_agents_guide.md)) |
| 7 | **Agents after eval only.** Tool-using loops come after Days 1–6 retrieval/faithfulness checks. Short reasoning-model note (when costly thinking APIs help). |

**Agents-after-eval rule:** Do not start Day 7 agent demos until you have a tiny retrieval/faithfulness eval from Day 3 and basic guardrails from Day 6. Sandbox tools first. Gate any write side effects.

### Check yourself

Wrong idea: “BLEU on chatbot replies proves the RAG works.”  
Better: score retrieval hit-rate and faithfulness on a fixed question set with known sources.  
Try this: write five questions with expected document IDs. Measure how often retrieval returns them before you tune prompts.

**After the spine:** Staging checklist in [resources/genai_production_deployment.md](../resources/genai_production_deployment.md).

## What You'll Learn

- Prompt engineering and schema-shaped structured outputs
- Embeddings and vector stores (FAISS, Chroma, managed options)
- Hybrid RAG, rerank pointers, faithfulness / retrieval eval
- Context engineering, caching, and cost basics
- When to stop at prompts / RAG vs try LoRA / QLoRA (PEFT)
- Guardrails, prompt injection awareness, MCP permissions
- Tool-using agents in sandboxes after eval
- Generative config parameters (temperature, top-p, and friends)

##  Topics Covered

### 1. Prompt Engineering
- **What is Prompt Engineering**: Designing effective inputs for LLMs
- **Zero-shot Prompting**: No examples, rely on pre-trained knowledge
- **Few-shot Prompting**: Provide examples to guide behavior
- **Chain-of-Thought**: Step-by-step reasoning
- **Generative Configuration**: Temperature, top-p, top-k, repetition penalty
- **Unsupported answers**: Mitigation strategies (RAG helps grounding; it does not fully eliminate errors)
- **Text Embeddings**: Vector representations for semantic search
- **Advanced Techniques**: Role-playing, output formatting, constraints

### 2. Vector Databases
- **What are Vector Databases**: Storage for high-dimensional embeddings
- **Pinecone**: Managed cloud vector database
- **ChromaDB**: Open-source, Python-first vector database
- **Weaviate**: GraphQL-based vector search engine
- **FAISS**: Facebook AI Similarity Search library
- **Semantic Search**: Finding similar documents by meaning
- **Similarity Metrics**: Cosine similarity, Euclidean distance
- **Choosing the Right Database**: Comparison and use cases

### 3. RAG (Retrieval-Augmented Generation)
- **RAG Architecture**: Retrieval + Augmentation + Generation
- **Document Ingestion**: Loading and processing documents
- **Text Chunking**: Strategies for splitting documents
- **Embedding Generation**: Creating vector representations
- **Vector Database Integration**: Storing and retrieving embeddings
- **Query Processing**: User query to embedding conversion
- **Context Augmentation**: Combining retrieved context with prompts
- **LLM Integration**: Generating responses with augmented context
- **Evaluation**: Faithfulness and retrieval metrics (not BLEU as the main score)
- **Staging patterns**: Caching, cost, and monitoring before a careful live path

### 4. LLM Agents (Day 7, after eval)
- **What are AI Agents**: Loops that plan, call tools, and observe results
- **LangChain Agents / LangGraph**: Learning tools for tool-using workflows
- **Tools and Function Calling**: Start read-only. Gate writes
- **Memory and Context**: Managing conversation history without stuffing the window
- **ReAct-style loops**: Reason and act with tools
- **Agent Evaluation**: Fixed task lists before you trust a demo

### 5. Multi-Agent Systems and MCP
- **Multi-Agent Architectures**: Optional demos when one loop is not enough
- **Specialized Roles**: Planner, research, writer patterns as teaching examples
- **CrewAI / AutoGen**: Framework options to explore, not required
- **MCP (Model Context Protocol)**: Tool and context access with permissions
- **Prompt injection**: Treat user and retrieved text as untrusted before side effects

### 6. Staging GenAI apps
- **Tech stack**: Frontend, backend, LLM APIs, vector stores
- **Streamlit / FastAPI**: Common demo and API shapes
- **Cost and caching**: Prefix / prompt cache ideas, batching, quantization concepts
- **Monitoring**: Latency, tokens, error rates in staging
- **Security**: API keys, input checks, rate limits
- **Lifecycle**: Problem definition, then demo, then eval, then careful live

## Learning Objectives

By the end of this module, you should be able to:
- Design prompts and schema-shaped structured outputs
- Set up embeddings and a vector store for semantic search
- Build a demo RAG loop with hybrid retrieval ideas and faithfulness checks
- Decide Prompt vs RAG vs LoRA / QLoRA with a short written rationale
- Add basic guardrails before tools that change state
- Run a tool-using agent in a sandbox after Days 1-6
- Apply cost and caching basics on a staging path

## Projects

1. **RAG document Q&A (demo)**: Ingest docs, store embeddings, answer with citations, score retrieval on a tiny set
2. **Prompt + schema playground**: Structured outputs with validation, not free-form JSON hopes
3. **Hybrid retrieval experiment**: BM25 + vectors (+ optional rerank). Compare hit rate
4. **Guarded tool agent (sandbox)**: Read-only tools first. Confirm before any write
5. **PEFT decision memo**: One page arguing Prompt, RAG, or LoRA for a sample task

## Key Concepts

- **Prompt engineering**: Clear instructions and schemas for models
- **Semantic search**: Finding information by meaning, not only keywords
- **RAG**: Retrieval plus generation for knowledge outside model weights
- **Faithfulness / retrieval eval**: Groundedness and hit rate. Not BLEU as the main RAG score
- **Agents**: Tool-using loops. Demos until eval and human gates exist
- **PEFT (LoRA / QLoRA)**: Efficient adaptation after Prompt and RAG fail your checks
- **Staging deployment**: Reliability, cost, and security before a careful live path

## Documentation & Learning Resources

**Official Documentation:**
- [LangChain Documentation](https://python.langchain.com/): Complete LangChain guide
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/): Graph-based workflows
- [Pinecone Documentation](https://docs.pinecone.io/): Managed vector database
- [ChromaDB Documentation](https://docs.trychroma.com/): Open-source vector database
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/): Pre-trained models
- [OpenAI API Documentation](https://platform.openai.com/docs): GPT models and API

**Free Courses:**
- [LangChain Crash Course](https://www.youtube.com/watch?v=lG7Uxts9SXs): Free YouTube course
- [RAG Tutorial (LangChain)](https://python.langchain.com/docs/use_cases/question_answering/): Free tutorial
- [Vector Databases Course](https://www.deeplearning.ai/short-courses/vector-databases/): DeepLearning.AI course
- [Building LLM Applications](https://www.deeplearning.ai/short-courses/building-applications-with-llms/): DeepLearning.AI course

**Tutorials:**
- [Prompt Engineering Guide](../resources/generative_ai_comprehensive_guide.md#prompt-engineering): Prompt patterns and failure modes
- [RAG Implementation Guide](../resources/rag_comprehensive_guide.md): RAG pipeline notes
- [LangChain Tutorial](../resources/langchain_guide.md): LangChain framework guide
- [AI Agents Guide](../resources/ai_agents_guide.md): Building AI agents
- [Vector Databases Guide](../resources/generative_ai_comprehensive_guide.md#vector-databases): Vector database comparison

**Video Tutorials:**
- [LangChain Crash Course (YouTube)](https://www.youtube.com/watch?v=lG7Uxts9SXs)
- [RAG Tutorial (YouTube)](https://www.youtube.com/watch?v=8OJC21T2SQ4)
- [Building LLM Apps (YouTube)](https://www.youtube.com/playlist?list=PLIUOU7oqGTLieV9uTfD-7qHO8zJqkRnZC)
- [Vector Databases Explained](https://www.youtube.com/watch?v=oZWVmJ5nP3U)

**Practice:**
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates): Example projects
- [RAG Examples](https://github.com/langchain-ai/langchain/tree/master/templates/rag): RAG implementations
- [Hugging Face Spaces](https://huggingface.co/spaces): Deploy and share GenAI apps
- [LangChain Playground](https://smith.langchain.com/): Experiment with LangChain

**[Complete Detailed Guide](generative-ai-llms.md)**

**Additional Resources:**
- [Advanced Topics](generative-ai-llms-advanced-topics.md): Advanced RAG, agent architectures, optimization
- [Project Tutorial](generative-ai-llms-project-tutorial.md): Step-by-step RAG system project
- [Quick Reference](generative-ai-llms-quick-reference.md): Quick lookup guide for GenAI
- [Generative AI Guide](../resources/generative_ai_comprehensive_guide.md): Concept overview
- [RAG Guide](../resources/rag_comprehensive_guide.md): Deeper RAG implementation notes
- [LangChain Guide](../resources/langchain_guide.md): LangChain framework details
- [AI Agents Guide](../resources/ai_agents_guide.md): AI agents and multi-agent systems
- [GenAI Production Deployment](../resources/genai_production_deployment.md): Production deployment patterns

---

**Previous Module:** [12-natural-language-processing](../12-natural-language-processing/README.md)  
**Next (recommended stage order):** [20-handling-imbalanced-data](../20-handling-imbalanced-data/README.md) (Stage 7.5), then [13-model-deployment](../13-model-deployment/README.md) (Stage 8)

**Folder order note:** Module 19 (SQL) is **Stage 1.5**, study it in parallel with Modules 01–02, not after this module. See [FOUNDATION_AND_JOB_READINESS.md](../FOUNDATION_AND_JOB_READINESS.md).
