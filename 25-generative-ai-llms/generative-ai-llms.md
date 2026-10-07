# Generative AI & Modern LLM Applications

This file is the **module hub** (map + links), not a full textbook chapter. Deep implementation lives under `resources/`. Work the **Day 1-7 study spine** below in order. Then return here for deeper links.

Treat early builds as demos or staging apps. Eval and guardrails come before tools that change state (send email, write files, call paid APIs).

## Table of Contents

- [Day 1-7 study spine](#day-1-7-study-spine)
- [Topic map (depth links)](#topic-map-depth-links)
- [Resources](#resources)

---

## Day 1-7 study spine

Follow this order. Skipping ahead to agents without retrieval eval is a common failure mode.

### Day 1. Prompts and structured outputs

**Focus:** Clear instructions, few-shot examples, and **schema-shaped outputs** (JSON Schema / tool schemas / typed fields). Prefer a declared schema over “please reply in JSON.”

**Do:**

- Write a task prompt with role, constraints, and one good example.
- Return a fixed shape (fields + types). Validate the parse before you trust the result.
- Try temperature and max tokens. Note how loose prompts break parsers.

**Depth:** [Prompt engineering](../resources/generative_ai_comprehensive_guide.md#prompt-engineering)

### Day 2. Embeddings and vector store basics

**Focus:** Chunk text, embed, store vectors, run similarity search. Know cosine vs Euclidean at a practical level.

**Do:**

- Chunk a small doc set (overlap is a choice you measure later).
- Index in a local store (FAISS / Chroma) or a managed one for a demo.
- Query with a few paraphrases. Note misses that keyword search would catch.

**Depth:** [Vector databases](../resources/generative_ai_comprehensive_guide.md#vector-databases)

### Day 3. Hybrid RAG and faithfulness / retrieval eval

**Focus:** Combine BM25 (or keyword) with vector search. Add a **rerank** step when you have candidates. Evaluate **retrieval** and **faithfulness** (does the answer stick to retrieved context?). Do **not** treat BLEU as your RAG quality metric.

**Do:**

- Build a thin RAG loop. Return citations or chunk ids with answers.
- Score retrieval (hit rate / recall@k on a tiny labeled set). Spot-check faithfulness by hand or with a groundedness check.
- Hybrid pointer: BM25 + vectors, then optional cross-encoder rerank. See [RAG guide](../resources/rag_comprehensive_guide.md) and Advanced Topics hybrid notes.

**Depth:** [RAG comprehensive guide](../resources/rag_comprehensive_guide.md) · [Advanced RAG](generative-ai-llms-advanced-topics.md#advanced-rag-techniques)

### Day 4. Context engineering and prompt caching / cost

**Focus:** What you put in context (system rules, retrieved chunks, history) drives quality and bill. Learn **prefix / prompt caching** ideas and basic inference cost levers.

**Do:**

- Trim history. Prefer stable system prefixes that can be cached when your provider supports it.
- Log tokens per request in a demo. Compare “stuff everything” vs ranked top-k chunks.
- Read the practical inference notes (batching, KV cache concept, quantization concept).

**Depth:** [GenAI production deployment](../resources/genai_production_deployment.md#context-and-prefix-caching) · [Inference basics](../resources/genai_production_deployment.md#inference-basics-batching-kv-cache-quantization)

### Day 5. PEFT decision path (Prompt, then RAG, then LoRA / QLoRA)

**Focus:** Decide **when not to fine-tune**. Default order: better prompts, then RAG / tools for facts, then PEFT (LoRA / QLoRA) only when style, format, or domain behavior still fails after eval.

**Do:**

- Write a one-page decision for a sample task. Facts that change? Prefer RAG. Tone or schema the base model already gets? Prefer prompts.
- Skim PEFT in NLP advanced topics. Then the fine-tuning guide decision section.

**Depth:** [PEFT in NLP advanced topics](../12-natural-language-processing/nlp-advanced-topics.md#parameter-efficient-fine-tuning-peft) · [Transformer fine-tuning guide (decision path)](../resources/transformer_fine_tuning_guide.md#peft-decision-path-prompt-then-rag-then-loraqlora)

### Day 6. Guardrails, prompt injection, MCP permissions

**Focus:** Before any tool with side effects, add input/output checks and treat retrieved or user text as **untrusted**. Learn MCP as a way to expose tools with **permissions**, not as “give the model everything.”

**Do:**

- List tools that only read vs tools that write or spend money. Gate writes behind confirmations in demos.
- Try a simple injection case (“ignore previous instructions…” in a retrieved doc). Log when the model complies.
- Skim MCP overview and permission ideas in the agents guide.

**Depth:** [AI agents guide](../resources/ai_agents_guide.md) · [Security notes in Advanced Topics](generative-ai-llms-advanced-topics.md#security-and-safety)

### Day 7. Agents last, plus a short reasoning-model note

**Focus:** Agents (plan, tool, observe) come **after** solid prompts, retrieval, eval, and guardrails. Multi-agent setups are optional demos, not a default production shape.

**Reasoning-model note:** Some APIs charge extra for longer “thinking” traces. Use them when hard multi-step problems fail under a normal call and your eval shows a clear gain. Skip them for simple extract / classify / RAG answer paths where cost rises without quality lift.

**Do:**

- One tool-using agent in a sandbox. Measure success on a fixed task list.
- Keep side-effect tools behind the Day 6 gates.
- Optional: query rewrite + doc grading (agentic RAG) in Advanced Topics.

**Depth:** [AI agents guide](../resources/ai_agents_guide.md) · [Agentic RAG survey](generative-ai-llms-advanced-topics.md#agentic-rag-survey)

---

## Topic map (depth links)

### Prompt engineering

- [Generative AI guide · Prompt engineering](../resources/generative_ai_comprehensive_guide.md#prompt-engineering)
- Zero-shot, few-shot, chain-of-thought
- Structured outputs via schema, not polite JSON requests
- Unsupported answers still happen. RAG helps grounding. It does not remove all errors.

### Vector databases

- [Generative AI guide · Vector databases](../resources/generative_ai_comprehensive_guide.md#vector-databases)
- Pinecone, ChromaDB, Weaviate, FAISS (pick one for learning)
- Semantic search and similarity metrics

### RAG systems

- [RAG comprehensive guide](../resources/rag_comprehensive_guide.md)
- Ingest, chunk, embed, retrieve, generate
- Hybrid search, rerank, faithfulness / retrieval eval (not BLEU as the main score)
- Multimodal / document-vision RAG survey in [Advanced Topics](generative-ai-llms-advanced-topics.md#multimodal-and-document-vision-rag)

### LLM agents (after Days 1-6)

- [AI agents guide](../resources/ai_agents_guide.md)
- Tools and function calling in a sandbox first
- LangChain / LangGraph style workflows as learning tools
- Treat “autonomous” demos as demos until eval and human gates exist

### Multi-agent and MCP

- [AI agents guide · Multi-agent](../resources/ai_agents_guide.md#multi-agent-systems)
- MCP as standardized tool / context access with permissions
- Specialized roles only when a single agent loop is not enough

### Staging and ops checklist

- [GenAI production deployment](../resources/genai_production_deployment.md)
- Cost, caching, monitoring, security for staging, then careful live paths
- Prefer demo language until you have eval, rate limits, and rollback

---

## Resources

### Comprehensive guides

- [Generative AI guide](../resources/generative_ai_comprehensive_guide.md)
- [RAG guide](../resources/rag_comprehensive_guide.md)
- [LangChain guide](../resources/langchain_guide.md)
- [AI agents guide](../resources/ai_agents_guide.md)
- [GenAI production deployment](../resources/genai_production_deployment.md)
- [Transformer fine-tuning guide](../resources/transformer_fine_tuning_guide.md)

### In this module folder

- [Advanced Topics](generative-ai-llms-advanced-topics.md)
- [Project tutorial](generative-ai-llms-project-tutorial.md)
- [Quick reference](generative-ai-llms-quick-reference.md)

---

**Next:** Run the project tutorial as a **demo RAG** with citations. Add faithfulness checks before you widen scope.

> [!RECALL]
> Why build a RAG system instead of relying on the LLM alone?
>
> RAG retrieves external documents so answers can use knowledge beyond the model's parameters. This module frames RAG as knowledge-augmented AI for document Q&A. Eval retrieval and faithfulness. Do not use BLEU as your main RAG score.
