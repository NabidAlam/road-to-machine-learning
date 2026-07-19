# AI Engineering Glossary

Practical definitions for modern AI engineering: LLMs, agents, RAG, training, serving, and production systems.

Each entry includes:

- **What people say:** the casual shorthand you hear in meetings and Discord
- **What it actually means:** the precise technical meaning

Pairs with the classic [ML Glossary](ml_glossary.md). Prefer this page for LLM, agent, and GenAI product work. When a term tempts you into a wrong story (understanding, autonomy, creativity), read [AI Myths Busted](ai_myths_busted.md).

## Table of Contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [J](#j)
- [K](#k)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [Q](#q)
- [R](#r)
- [S](#s)
- [T](#t)
- [U](#u)
- [V](#v)
- [W](#w)
- [Z](#z)
- [How to use this glossary](#how-to-use-this-glossary)

---

## A

### Activation Function

**What people say:** "The nonlinear thing between layers."

**What it actually means:** A function applied after each linear layer that introduces nonlinearity. Without it, stacking any number of linear layers collapses into a single linear transformation. Common examples: ReLU, GELU, and SiLU. The choice directly affects how gradients flow during training.

### Adam (Optimizer)

**What people say:** "The default optimizer."

**What it actually means:** Adaptive Moment Estimation. It combines momentum (first moment) with adaptive learning rates per parameter (second moment). It includes bias correction for early training steps and works robustly across most tasks without extensive tuning.

### AdamW

**What people say:** "Adam but better."

**What it actually means:** Adam with decoupled weight decay. In standard Adam, L2 regularization is scaled by the adaptive learning rate. Which is often suboptimal. AdamW applies weight decay directly to the weights, independent of gradient statistics. It is the industry standard for training transformers.

### Agent

**What people say:** "An autonomous AI that thinks and acts on its own."

**What it actually means:** A control loop where an LLM decides which tool to call, executes it, observes the result, and repeats until the goal is met.

**Why it's called that:** Borrowed from philosophy: an "agent" acts in the world. In AI engineering it is simply LLM + tools + loop.

### Agentic RAG

**What people say:** "RAG that can think and search again."

**What it actually means:** A retrieval pattern where the model decides when to retrieve, rewrite the query, fetch more documents, or verify an answer. That is different from a single fixed retrieve-then-generate pass.

### Alignment

**What people say:** "Making AI safe."

**What it actually means:** The technical challenge of ensuring an AI system's behavior matches human intentions, values, and preferences, especially regarding edge cases the designer did not anticipate.

### Attention

**What people say:** "How the AI focuses on important parts."

**What it actually means:** A mechanism where every token computes a weighted sum of all other tokens' values, with weights determined by relevance (via the dot product of query and key vectors).

### Autograd

**What people say:** "Automatic gradients."

**What it actually means:** A system that records operations on tensors and computes gradients via reverse-mode differentiation. PyTorch builds dynamic computation graphs on-the-fly, while JAX uses function transformations (like `grad`). This makes backpropagation practical by automating the calculation of derivatives.

### Autoregressive

**What people say:** "The AI generates one word at a time."

**What it actually means:** A model that predicts the next token conditioned on all previous tokens, then feeds that prediction back as input for the next step. GPT, LLaMA, and Claude are all autoregressive.

---

## B

### Backpropagation

**What people say:** "How neural networks learn."

**What it actually means:** An algorithm that computes how much each weight contributed to the total error by applying the chain rule backward through the network, then adjusting weights proportionally.

### Batch Size

**What people say:** "How many examples at once."

**What it actually means:** The number of training examples processed in one forward/backward pass before updating weights. Larger batches provide more stable gradient estimates but consume more memory. Batch size interacts with the learning rate (the "linear scaling rule").

### BM25

**What people say:** "Old-school keyword search."

**What it actually means:** A sparse ranking function used in classic information retrieval. It scores documents by term frequency and inverse document frequency with length normalization. Still a strong baseline and often combined with vector search in hybrid retrieval.

---

## C

### Chain of Thought (CoT)

**What people say:** "Making the AI think step by step."

**What it actually means:** A prompting technique where the model is asked to show its reasoning steps. This improves accuracy on multi-step problems because each reasoning step conditions the next token generation.

### Chat Template

**What people say:** "How the model expects messages."

**What it actually means:** The exact formatting that turns role-tagged messages (`system`, `user`, `assistant`, `tool`) into the token sequence the model was trained on. Wrong templates silently degrade quality.

### Chunking

**What people say:** "Splitting documents into pieces."

**What it actually means:** Breaking text into segments before embedding for retrieval. Chunk size determines the granularity of search results. Common strategies: fixed-size with overlap, sentence-based, or semantic splitting.

### CNN (Convolutional Neural Network)

**What people say:** "Image AI."

**What it actually means:** A network that uses convolution operations (sliding filters over input) to detect local patterns. Stacking convolutions detects increasingly complex features, from edges and textures to full objects.

### Constrained Decoding

**What people say:** "Force the model to output valid JSON."

**What it actually means:** Restricting next-token choices at generation time so outputs obey a grammar or schema (for example JSON Schema). Reduces parse failures for tool calling and structured APIs.

### Context Window

**What people say:** "How much the model can remember at once."

**What it actually means:** The maximum number of tokens the model can attend to in one forward pass (prompt + generated tokens). Exceeding it requires truncation, summarization, or a long-context architecture.

### Contrastive Learning

**What people say:** "Learning by comparison."

**What it actually means:** Training by pulling similar pairs closer together and pushing dissimilar pairs apart in embedding space (for example CLIP matching image-text pairs).

### Cosine Similarity

**What people say:** "How similar two vectors are."

**What it actually means:** The cosine of the angle between two vectors. It ranges from -1 (opposite) to 1 (identical direction), ignoring magnitude. This is the standard metric for embeddings and semantic search.

### Cross-Encoder Reranker

**What people say:** "The second search pass that makes results better."

**What it actually means:** A model that scores a query and a candidate document together (not as separate embeddings). Slower than bi-encoders, but usually more accurate for the top-k shortlist.

### Cross-Entropy

**What people say:** "The classification loss."

**What it actually means:** A measure of the difference between two probability distributions. In language models, it represents the negative log probability of the correct next token. Lower is better; perplexity is simply the exponential of cross-entropy.

### CUDA

**What people say:** "GPU programming."

**What it actually means:** NVIDIA's parallel computing platform. It allows matrix operations to run on thousands of GPU cores simultaneously. PyTorch and TensorFlow use CUDA under the hood.

---

## D

### Data Augmentation

**What people say:** "Making more training data."

**What it actually means:** Creating modified copies of existing data (rotating images, paraphrasing text) to increase training set diversity and reduce overfitting without collecting new data.

### Decoder

**What people say:** "The output part."

**What it actually means:** In transformers, a decoder uses causal (masked) self-attention, meaning each position can only attend to earlier positions. GPT is decoder-only.

### Diffusion Model

**What people say:** "AI that generates images from noise."

**What it actually means:** A model trained to reverse a gradual noising process. It learns to predict and remove noise; at generation time, it starts from pure noise and iteratively denoises it into a coherent output.

### Distillation

**What people say:** "Teaching a small model from a big one."

**What it actually means:** Training a smaller student model to match a larger teacher model's outputs or internal signals. Used to cut latency and cost while keeping much of the teacher's behavior.

### DPO (Direct Preference Optimization)

**What people say:** "A simpler RLHF."

**What it actually means:** A training method that skips the separate reward model entirely. It optimizes the language model directly to prefer the better response in pairs of human-labeled preferences.

### Dropout

**What people say:** "Randomly turning off neurons."

**What it actually means:** During training, randomly setting a fraction of activations to zero. This forces the network to avoid relying on any single neuron, acting as a simple but effective regularization technique.

---

## E

### Eigenvalue

**What people say:** "Some math thing for PCA."

**What it actually means:** For a matrix \(A\), an eigenvalue \(\lambda\) satisfies \(Av = \lambda v\). It indicates how much the matrix scales vectors in that direction. Large eigenvalues correspond to directions of high variance in the data.

### Embedding

**What people say:** "Some AI magic that turns words into numbers."

**What it actually means:** A learned mapping from discrete items (words, images) to dense vectors in a continuous space, where similar items end up geometrically close together.

### Encoder

**What people say:** "The input part."

**What it actually means:** In transformers, an encoder uses bidirectional self-attention, allowing each position to attend to all others. BERT is encoder-only; it is ideal for understanding tasks (classification, NER) but not generation.

### Epoch

**What people say:** "One pass through the data."

**What it actually means:** A complete pass through every example in the training set. Multiple epochs allow the model to learn patterns more deeply but risk overfitting.

### Eval / Evaluation Harness

**What people say:** "Tests for the model."

**What it actually means:** A fixed set of tasks, datasets, and metrics used to measure model or prompt quality over time. In production AI, evals are how you catch regressions before shipping prompt or model changes.

---

## F

### Feature

**What people say:** "A column in your data."

**What it actually means:** An individual measurable property of the data. In deep learning, the network learns these features automatically from raw data rather than requiring manual engineering.

### Few-Shot

**What people say:** "Give the AI some examples first."

**What it actually means:** Including a small number of input-output examples in the prompt to allow the model to pattern-match the desired format and behavior.

### Fine-tuning

**What people say:** "Training the AI on your data."

**What it actually means:** Starting with a pre-trained model's weights and continuing training on a smaller, task-specific dataset. It adapts existing knowledge rather than learning from scratch.

### Function Calling

**What people say:** "AI that can use tools."

**What it actually means:** A structured way for LLMs to request execution of external functions defined by JSON Schemas. The model outputs a JSON object specifying the function and arguments, which the host application then executes.

---

## G

### GAN (Generative Adversarial Network)

**What people say:** "Two AIs fighting each other."

**What it actually means:** A generator network tries to create realistic data while a discriminator network tries to distinguish real from fake. They train simultaneously; as the generator improves, the discriminator must also get better, driving high-fidelity generation.

### Golden Dataset

**What people say:** "The ground-truth test set for prompts."

**What it actually means:** A curated set of inputs with expected answers or rubrics used for regression testing of prompts, tools, and RAG pipelines.

### Gradient

**What people say:** "The slope."

**What it actually means:** A vector of partial derivatives pointing in the direction of steepest increase. In ML, you move opposite to the gradient (gradient descent) to minimize the loss.

### Gradient Descent

**What people say:** "How AI improves."

**What it actually means:** An optimization algorithm that adjusts model parameters to reduce the loss function, effectively navigating a high-dimensional landscape toward a minimum.

### GPT

**What people say:** "ChatGPT or the AI."

**What it actually means:** Generative Pre-trained Transformer. Generative (produces text), Pre-trained (trained on large corpora), Transformer (the underlying architecture).

### GraphRAG

**What people say:** "RAG with a knowledge graph."

**What it actually means:** Retrieval that uses entities and relationships (a graph) in addition to or instead of pure vector chunks, often improving multi-hop and relational questions.

### Guardrails

**What people say:** "Safety filters for AI."

**What it actually means:** Input/output validation layers that detect and block harmful content, prompt injection, or PII leakage. They can be rule-based (regex) or model-based (classifiers).

---

## H

### Hallucination

**What people say:** "The AI is lying."

**What it actually means:** The model generates plausible-sounding but factually incorrect text. It is a result of the model pattern-completing rather than retrieving facts from a database.

### Human-in-the-Loop (HITL)

**What people say:** "A human has to approve."

**What it actually means:** Workflow design where critical agent actions pause for human review before execution. Examples include sending emails, spending money, or changing production systems.

### Hybrid Search

**What people say:** "Keyword search plus vector search."

**What it actually means:** Combining sparse retrieval (BM25) with dense embedding search, then fusing ranks (often Reciprocal Rank Fusion). Usually more robust than either method alone.

### Hyperparameter

**What people say:** "Settings you tune."

**What it actually means:** Values set before training (for example learning rate, batch size, dropout rate) that control the learning process itself, rather than being learned from data.

### HyDE

**What people say:** "Search with a fake answer first."

**What it actually means:** Hypothetical Document Embeddings. The model writes a plausible answer, you embed that, and retrieve real documents similar to the hypothetical answer. This often improves recall for short queries.

---

## I

### In-Context Learning

**What people say:** "The model learns from the prompt."

**What it actually means:** The ability of LLMs to adapt behavior from examples or instructions inside the context window without updating weights.

### Inference

**What people say:** "Running the AI."

**What it actually means:** Using a trained model to make predictions on new data. No weight updates occur during this phase.

### Inductive Bias

**What people say:** "Never heard of it."

**What it actually means:** The assumptions built into a model's architecture (for example CNNs assume local patterns matter). Proper inductive bias helps models learn faster with less data.

### Instruction Tuning

**What people say:** "Teaching the model to follow instructions."

**What it actually means:** Supervised fine-tuning on (instruction, response) pairs so a base language model becomes a helpful assistant that follows user intents.

---

## J

### JAX

**What people say:** "Google's ML framework."

**What it actually means:** A NumPy-compatible library for high-performance ML research. It supports automatic differentiation (`grad`), JIT compilation (`jit`), and vectorization (`vmap`). It follows a purely functional programming style.

### Jailbreak

**What people say:** "Tricks to make the AI ignore its rules."

**What it actually means:** Adversarial prompts designed to bypass safety policies or system instructions. Related to prompt injection, but usually aimed at policy refusal rather than tool abuse.

---

## K

### KV Cache

**What people say:** "Makes inference faster."

**What it actually means:** During autoregressive generation, caching the key and value matrices from previous tokens to avoid recomputing them at each step. This trades memory for significantly faster inference.

---

## L

### Latent Space

**What people say:** "The hidden representation."

**What it actually means:** A compressed, learned representation space where similar inputs map to nearby points. It is lower-dimensional than the input but captures the fundamental structure of the data.

### Learning Rate

**What people say:** "How fast the AI learns."

**What it actually means:** A scalar controlling the step size during gradient descent. It is arguably the most critical hyperparameter to tune.

### LLM (Large Language Model)

**What people say:** "AI or the brain."

**What it actually means:** A transformer-based neural network trained to predict the next token, characterized by billions of parameters and training on internet-scale text.

### LLM-as-Judge

**What people say:** "Using AI to grade AI."

**What it actually means:** Evaluating outputs with another LLM against a rubric. Useful at scale, but judges can be biased. Calibrate against human labels when stakes are high.

### LoRA (Low-Rank Adaptation)

**What people say:** "Efficient fine-tuning."

**What it actually means:** A method that freezes the main model weights and injects small, trainable low-rank matrices. This reduces memory requirements for fine-tuning by 10x–100x.

### Loss Function

**What people say:** "How wrong the AI is."

**What it actually means:** A function measuring the gap between predicted and actual output. Training is the process of minimizing this value.

### Lost-in-the-Middle

**What people say:** "The model ignores the middle of long prompts."

**What it actually means:** Empirically, many LLMs attend more strongly to the beginning and end of long contexts than to the middle. Important when placing retrieved documents in a RAG prompt.

---

## M

### MCP (Model Context Protocol)

**What people say:** "A way for AI to use tools."

**What it actually means:** An open standard (JSON-RPC) that allows AI applications to connect to external data sources and tools in a standardized, interoperable way.

### Mixed Precision

**What people say:** "Training trick for speed."

**What it actually means:** Using float16 for most operations (speed/memory) while maintaining float32 for sensitive weight updates (precision).

### Model Routing

**What people say:** "Pick the cheap model when possible."

**What it actually means:** Sending each request to an appropriate model (small vs large, fast vs accurate) based on complexity, cost, or confidence. This is a core production pattern for cost control.

### MoE (Mixture of Experts)

**What people say:** "Only part of the model runs."

**What it actually means:** A model architecture where a router sends each input to only a subset of expert subnetworks. This allows for massive parameter counts while keeping the compute cost per token low.

### Multimodal

**What people say:** "AI that sees images too."

**What it actually means:** Models that accept or produce more than one modality (text, image, audio, video). Vision-language models are the most common example in product AI today.

---

## N

### NaN (Not a Number)

**What people say:** "Training crashed."

**What it actually means:** A floating-point value indicating an undefined calculation (for example division by zero). In training, this usually signals exploding gradients or improper learning rates.

### Normalization

**What people say:** "Scaling the data."

**What it actually means:** Adjusting values to a standard range to stabilize training. Techniques like Layer Normalization are critical for the stability of modern transformers.

---

## O

### Observability (LLM Apps)

**What people say:** "Logging for AI."

**What it actually means:** Tracing prompts, tool calls, retrieved chunks, latency, token usage, and failure modes so you can debug and improve production AI systems.

### Optimizer

**What people say:** "The thing that updates weights."

**What it actually means:** The algorithm (like Adam or SGD) that uses gradients to determine how to update model parameters.

### Overfitting

**What people say:** "The model memorized the data."

**What it actually means:** The model performs perfectly on training data but fails to generalize to unseen data. It has learned the noise instead of the signal.

---

## P

### Parameter

**What people say:** "Model size."

**What it actually means:** A learnable value (weight or bias) within the model. "7B parameters" means 7 billion numbers that define the model's behavior.

### PEFT

**What people say:** "Cheap fine-tuning methods."

**What it actually means:** Parameter-Efficient Fine-Tuning. A family of techniques (LoRA, adapters, prompt tuning) that train a small subset of parameters instead of the full model.

### Perplexity

**What people say:** "How confused the model is."

**What it actually means:** The exponential of the cross-entropy loss. Lower values indicate higher confidence and better model performance.

### Prefill vs Decode

**What people say:** "Why the first token is slow."

**What it actually means:** Prefill processes the entire prompt in parallel; decode generates tokens one by one using the KV cache. Latency and GPU utilization behave differently in each phase.

### Precision & Recall

**What people say:** "Accuracy metrics."

**What it actually means:** Precision measures the accuracy of positive predictions (how many flagged items were correct). Recall measures the ability to find all positive instances (how many correct items were caught).

### Prefix Caching

**What people say:** "Reuse the system prompt compute."

**What it actually means:** Caching KV states for shared prompt prefixes across requests so repeated system instructions and tool schemas do not get recomputed every time.

### Prompt Engineering

**What people say:** "Talking to AI the right way."

**What it actually means:** The practice of designing input text, including system instructions and few-shot examples, to reliably steer the model toward specific outputs.

### Prompt Injection

**What people say:** "Hacking the AI with words."

**What it actually means:** A security vulnerability where malicious input overrides the intended system instructions. Indirect injection hides instructions inside retrieved documents or tool outputs.

---

## Q

### QLoRA

**What people say:** "LoRA but cheaper."

**What it actually means:** Quantized LoRA. It keeps the base model in 4-bit precision while training LoRA adapters in higher precision, allowing massive models to be fine-tuned on consumer-grade hardware.

### Quantization

**What people say:** "Making the model smaller."

**What it actually means:** Reducing the precision of weights (for example from 32-bit to 8-bit or 4-bit) to reduce memory usage and speed up inference with minimal accuracy loss.

---

## R

### RAG (Retrieval-Augmented Generation)

**What people say:** "AI that can search."

**What it actually means:** A pattern where the model retrieves relevant documents from an external knowledge base, injects them into the prompt, and generates an answer grounded in that information.

### ReAct

**What people say:** "Reason then act."

**What it actually means:** An agent prompting pattern that interleaves reasoning traces with tool actions (`Thought → Action → Observation`), improving multi-step tool use.

### Reciprocal Rank Fusion (RRF)

**What people say:** "Merge two ranked lists."

**What it actually means:** A simple rank fusion formula that combines results from multiple retrievers without needing calibrated scores. Common in hybrid BM25 + vector search.

### Red Teaming

**What people say:** "Attack your own AI on purpose."

**What it actually means:** Systematic adversarial testing of prompts, tools, and policies to find jailbreaks, injections, and unsafe behaviors before attackers do.

### RLHF (Reinforcement Learning from Human Feedback)

**What people say:** "How they make AI helpful."

**What it actually means:** A three-stage training pipeline: collect human preference data, train a reward model to mimic those preferences, and use reinforcement learning to align the LLM with those preferences.

### ReLU

**What people say:** "Activation function."

**What it actually means:** Rectified Linear Unit (\(f(x) = \max(0, x)\)). It is the most common activation function because it is computationally efficient and helps mitigate the vanishing gradient problem.

### ROUGE

**What people say:** "Summarization metric."

**What it actually means:** A metric measuring the overlap between generated text and reference text. It is cheap to compute but primarily measures surface-level similarity rather than semantic meaning.

---

## S

### Scaling Laws

**What people say:** "Bigger is better."

**What it actually means:** Empirical relationships showing how loss improves as you scale model size, dataset size, and compute. Useful for planning training runs; not a guarantee of every capability.

### Semantic Search

**What people say:** "Smart search that understands meaning."

**What it actually means:** Searching by vector similarity rather than keyword matching. By mapping queries and documents into the same embedding space, it can match concepts (for example "payment failed" and "transaction declined").

### Self-Attention

**What people say:** "How the model decides what to focus on."

**What it actually means:** The core mechanism of transformers where each token calculates its relationship to every other token in the sequence via query, key, and value vectors.

### Serving Stack

**What people say:** "How we host the model."

**What it actually means:** The inference server and scheduling system that loads weights, batches requests, manages KV cache, and streams tokens (examples: vLLM, TensorRT-LLM, Hugging Face TGI).

### SFT (Supervised Fine-Tuning)

**What people say:** "Teaching the model to follow instructions."

**What it actually means:** Fine-tuning a base model on specific (instruction, response) pairs to shape its behavior into a conversational or task-oriented assistant.

### Softmax

**What people say:** "Turns numbers into probabilities."

**What it actually means:** A function that converts a vector of arbitrary numbers into a probability distribution that sums to 1.

### Speculative Decoding

**What people say:** "Draft with a small model, verify with a big one."

**What it actually means:** An inference speedup where a cheaper draft model proposes several tokens and the large model verifies them in parallel, accepting matching prefixes.

### Streaming

**What people say:** "Seeing the response appear word by word."

**What it actually means:** A method of sending tokens to the user as they are generated rather than waiting for the entire sequence to finish, which drastically improves perceived latency.

### Structured Output

**What people say:** "Make the model return JSON."

**What it actually means:** Generating responses that conform to a schema so downstream code can parse them reliably. Often implemented with constrained decoding or strict tool schemas.

### Swarm

**What people say:** "A bunch of AI agents working together like bees."

**What it actually means:** A multi-agent coordination pattern where agents share state and collaborate through message passing, leading to complex emergent behavior.

### System Prompt

**What people say:** "The AI's instructions."

**What it actually means:** A special instruction set provided at the start of a session that defines the model's persona, constraints, and behavior. It is distinct from user-provided prompts.

---

## T

### Temperature

**What people say:** "Creativity setting."

**What it actually means:** A hyperparameter that scales the logits before the softmax operation. Higher temperature flattens the distribution (more randomness/creativity), while lower temperature sharpens it (more deterministic/focused).

### Tensor

**What people say:** "A multi-dimensional array."

**What it actually means:** The fundamental data structure in deep learning (a generalization of scalars, vectors, and matrices). Tensors track computation history for autograd and are the objects that flow through a neural network.

### Token

**What people say:** "A word."

**What it actually means:** A subword unit (usually a few characters) generated by a tokenizer (for example BPE). LLMs process tokens, not full words.

### Token Budget

**What people say:** "How many tokens we can afford."

**What it actually means:** The practical limit for prompt + completion under cost, latency, and context-window constraints. Production systems allocate budget across system prompt, retrieved context, history, and answer.

### Top-p (Nucleus Sampling)

**What people say:** "Another creativity knob."

**What it actually means:** Sampling from the smallest set of tokens whose cumulative probability exceeds \(p\). Often preferred over raw temperature-only sampling for controllable randomness.

### Transfer Learning

**What people say:** "Using a pre-trained model."

**What it actually means:** Applying knowledge gained from one task (for example language modeling) to a different, downstream task (for example sentiment analysis).

### Transformer

**What people say:** "The architecture behind modern AI."

**What it actually means:** A neural network architecture that relies entirely on self-attention to process sequential data, allowing for massively parallel training compared to older recurrent models.

### TTFT (Time to First Token)

**What people say:** "How long until the reply starts."

**What it actually means:** Latency from request start until the first output token streams. Critical UX metric for chat products; dominated by prefill and queueing.

---

## U

### Underfitting

**What people say:** "The model isn't learning."

**What it actually means:** The model is too simple to capture the underlying data patterns. This results in high training and validation loss.

---

## V

### VAE (Variational Autoencoder)

**What people say:** "A generative model."

**What it actually means:** An autoencoder that forces its latent space to follow a probability distribution, allowing for the sampling and generation of new, realistic data points.

### Vector Database

**What people say:** "A special database for AI."

**What it actually means:** A database designed to store high-dimensional vectors and perform fast, approximate nearest-neighbor searches. Which is the foundational technology for RAG.

### Vision-Language Model (VLM)

**What people say:** "AI that understands images and text."

**What it actually means:** A multimodal model that jointly processes visual and textual inputs (for example image captioning, visual Q&A, document understanding).

---

## W

### Weight

**What people say:** "What the model learned."

**What it actually means:** A single numeric value in the model's parameter matrix that is adjusted during training to minimize loss.

### Weight Decay

**What people say:** "Regularization."

**What it actually means:** A technique that adds a penalty proportional to the magnitude of weights to the loss function, effectively preventing the weights from becoming excessively large.

---

## Z

### Zero-Shot

**What people say:** "No training needed."

**What it actually means:** Asking a model to perform a task without providing any task-specific examples in the prompt, relying entirely on the model's pre-existing knowledge.

---

## How to use this glossary

1. Skim letter sections when a meeting drops unfamiliar jargon.
2. Cross-check deeper guides when you need implementation detail:
   - [RAG Comprehensive Guide](rag_comprehensive_guide.md)
   - [AI Agents Guide](ai_agents_guide.md)
   - [Generative AI Comprehensive Guide](generative_ai_comprehensive_guide.md)
   - [Transformer Fine-Tuning Guide](transformer_fine_tuning_guide.md)
   - [ML Glossary](ml_glossary.md) for classic ML vocabulary
   - [AI Myths Busted](ai_myths_busted.md) when a definition fights a popular misconception
3. For Study Hub readers: open this guide on site, mark confidence as you go, and link terms back to module lessons when you meet them in projects.
