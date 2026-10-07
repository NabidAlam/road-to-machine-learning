# Project 4: LLM Chatbot & RAG System

Build a document Q&A chatbot with retrieval-augmented generation (RAG), citations, and a small evaluation set. A seq2seq-only dialog model without retrieval is a different exercise. Keep that as an optional baseline, not the main deliverable.

## Difficulty
Advanced

## Time Estimate
2-3 weeks

## Skills You'll Practice
- Prompt engineering
- Embeddings and vector stores
- RAG pipelines
- Retrieval and answer evaluation
- Demo or staging API / UI

## Learning Objectives

- Ingest and chunk a public or self-owned document corpus
- Embed chunks and retrieve top-k context for each query
- Generate answers grounded in retrieved context, with source citations
- Build a small eval set and report retrieval hit-rate plus faithfulness / groundedness
- Reduce unsupported answers with grounding rules and refusals when context is thin
- Ship a local or staging demo (API or simple UI). Do not claim a live production chatbot

## Dataset

Public docs (Python, FastAPI, scikit-learn docs, OWASP notes) or this curriculum’s own markdown. Optional dialog baseline only: [Cornell Movie Dialog](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

## Project Steps

1. Choose a GDPR-safe corpus (public docs or this repo’s markdown). No private company dumps.
2. Chunk documents and store embeddings in FAISS, Chroma, or similar.
3. Wire query → retrieve top-k → prompt with context → LLM answer + citations.
4. Create 30–80 eval questions answerable from the corpus (include multi-chunk and “should refuse” cases).
5. Score **primary** RAG quality: retrieval hit-rate, faithfulness / groundedness, citation coverage. Treat BLEU/ROUGE as optional secondary checks only.
6. Add basic guardrails (answer only from context, refuse when evidence is missing).
7. Demo locally or on staging (FastAPI, Streamlit, or Hugging Face Spaces). Log latency and token cost for a short sample of queries.

## Acceptance checklist

- [ ] README states corpus, chunking, model, and limits
- [ ] Eval set checked into the repo (questions + expected sources or answers)
- [ ] Reported hit-rate and at least one faithfulness / groundedness measure
- [ ] Sample answers show citations
- [ ] Deploy language is demo or staging, not “production live”

## Resources

- Module index brief: [Advanced projects README · Project 4](../README.md#project-4-llm-chatbot--rag-system)
- [Module 25: Generative AI & LLMs](../../25-generative-ai-llms/README.md)
- [RAG comprehensive guide](../../resources/rag_comprehensive_guide.md)
- [Capstone: LLM/RAG Engineer](../capstones/capstone-llm-rag-engineer.md)
- Optional seq2seq baseline: [Transformer tutorial (TensorFlow)](https://www.tensorflow.org/text/tutorials/transformer)
