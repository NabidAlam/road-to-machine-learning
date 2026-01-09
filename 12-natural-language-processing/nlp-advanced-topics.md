# Advanced Natural Language Processing Topics

Comprehensive guide to advanced NLP techniques and architectures.

## Table of Contents

- [Advanced Transformer Architectures](#advanced-transformer-architectures)
- [Training LLMs from Scratch](#training-llms-from-scratch)
- [Scaling LLMs for Production](#scaling-llms-for-production)
- [Mixture-of-Experts (MoE) Architecture](#mixture-of-experts-moe-architecture)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [Reward Modeling for RLHF](#reward-modeling-for-rlhf)
- [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
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

## Training LLMs from Scratch

### Why Train from Scratch?

**When to Train from Scratch:**
- **Custom Domain**: Specialized vocabulary or domain-specific knowledge
- **Research**: Understanding model internals and behavior
- **Resource Constraints**: Need smaller, efficient models
- **Learning**: Deep understanding of transformer architecture

**Challenges:**
- **Computational Cost**: Requires significant GPU resources
- **Data Requirements**: Need large, high-quality datasets
- **Hyperparameter Tuning**: Many parameters to optimize
- **Training Stability**: Requires careful initialization and learning rate scheduling

### Computational Challenges of Training LLMs

**The Scale Problem:**
Training modern LLMs requires enormous computational resources. Understanding these challenges is crucial for planning and optimization.

**1. Memory Constraints:**
- **Model Size**: A 7B parameter model requires ~14GB in FP32, ~7GB in FP16
- **Activation Memory**: Forward pass activations can exceed model size
- **Gradient Memory**: Backward pass stores gradients for all parameters
- **Total Memory**: Often 3-4x model size during training

**2. Compute Requirements:**
- **Training Time**: Weeks to months on hundreds of GPUs
- **FLOPS**: Trillions of floating-point operations per second
- **Cost**: Millions of dollars in compute for large models

**3. Data Pipeline Bottlenecks:**
- **Data Loading**: Must keep GPUs fed with data
- **Preprocessing**: Tokenization, batching, shuffling
- **Storage**: Terabytes of training data

**4. Distributed Training Complexity:**
- **Synchronization**: Gradient synchronization across GPUs
- **Communication Overhead**: Network bandwidth limitations
- **Fault Tolerance**: Handling GPU/node failures

**Solutions:**
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16/BF16 to reduce memory
- **Model Parallelism**: Split model across GPUs
- **Data Parallelism**: Replicate model, split data
- **Pipeline Parallelism**: Split layers across GPUs
- **ZeRO Optimization**: Partition optimizer states

### Scaling Laws for LLMs

**Empirical Scaling Laws:**
Research (Kaplan et al., 2020; Hoffmann et al., 2022) has revealed predictable relationships between model size, data, and compute.

**Key Findings:**

1. **Power Law Scaling:**
   - Loss decreases as a power law with model size, data, and compute
   - Formula: `L(N, D) = (N_c / N)^α_N + (D_c / D)^α_D`
   - Where N = parameters, D = data tokens, α ≈ 0.076

2. **Chinchilla Scaling (Compute-Optimal):**
   - Optimal data-to-parameter ratio: ~20 tokens per parameter
   - Example: 7B model needs ~140B tokens
   - Many models are "undertrained" (too few tokens)

3. **Emergent Abilities:**
   - Capabilities emerge at certain scales
   - Examples: Few-shot learning, reasoning, code generation
   - Not predictable from smaller models

**Practical Implications:**
```python
# Estimate compute requirements
def estimate_training_compute(model_params, data_tokens, flops_per_token=6):
    """
    Estimate training compute in FLOPs
    
    Args:
        model_params: Number of model parameters
        data_tokens: Number of training tokens
        flops_per_token: FLOPs per token (typically 6 for attention)
    
    Returns:
        Total FLOPs required
    """
    # Forward pass: 2 * params * tokens
    forward_flops = 2 * model_params * data_tokens
    
    # Backward pass: ~2x forward (gradients + activations)
    backward_flops = 2 * forward_flops
    
    # Total
    total_flops = forward_flops + backward_flops
    
    return total_flops

# Example: 7B model, 140B tokens
compute = estimate_training_compute(7e9, 140e9)
print(f"Total FLOPs: {compute:.2e}")
print(f"GPU-days (A100): {compute / (312e15 * 86400):.0f}")  # A100: 312 TFLOPS
```

**Scaling Strategy:**
- **Start Small**: Validate approach on small models
- **Scale Data**: Ensure sufficient training tokens
- **Scale Model**: Increase parameters for capacity
- **Monitor**: Track loss curves, emergent abilities
- **Budget**: Balance model size, data, and compute

### Domain Adaptation for Pre-training

**Domain-Specific Pre-training:**
Adapting pre-trained models to specific domains (e.g., finance, medicine, code) through continued pre-training.

**Why Domain Adaptation?**
- **Vocabulary**: Domain-specific terminology
- **Style**: Different writing conventions
- **Knowledge**: Domain-specific facts and relationships
- **Performance**: Better downstream task performance

**Approach:**
```python
# Continue pre-training on domain data
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare domain-specific corpus
domain_texts = load_domain_corpus()  # e.g., medical papers, financial reports

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = domain_texts.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./domain_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,  # Lower than initial pre-training
    warmup_steps=1000,
    logging_steps=100,
    save_steps=5000,
    fp16=True,  # Use mixed precision
    dataloader_num_workers=4
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

**Best Practices:**
- **Learning Rate**: 5-10x lower than initial pre-training
- **Data Quality**: High-quality, domain-specific corpus
- **Data Size**: Typically 10-100GB of domain text
- **Monitoring**: Watch for catastrophic forgetting
- **Evaluation**: Test on domain-specific benchmarks

### Building a Tiny LLM

**Step 1: Define Model Architecture**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyLLM(nn.Module):
    """A minimal transformer-based language model"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = x.size()
        
        # Create position indices
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create attention mask (1 for valid, 0 for padding)
        if mask is None:
            mask = torch.ones(batch_size, seq_length, device=x.device, dtype=torch.bool)
        
        # Convert to attention mask format (False for valid, True for padding)
        attn_mask = ~mask
        
        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

**Step 2: Data Preparation**

```python
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        for text in texts:
            # Tokenize and encode
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            })
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]

# Load and prepare data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Example: Load your text data
texts = ["Your training text here...", "More text..."]
train_dataset = TextDataset(texts, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

**Step 3: Training Loop**

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_llm(model, train_loader, num_epochs=10, learning_rate=3e-4, device='cuda'):
    """Train a language model"""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Create labels (shift by 1 for next token prediction)
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            
            # Forward pass
            logits = model(input_ids, mask=attention_mask)
            
            # Calculate loss (only on non-padding tokens)
            logits = logits[:, :-1, :].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Initialize and train
model = TinyLLM(
    vocab_size=len(tokenizer),
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048
)

train_llm(model, train_loader, num_epochs=10)
```

**Step 4: Text Generation**

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda'):
    """Generate text using the trained model"""
    
    model.eval()
    model = model.to(device)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(generated)
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

# Generate text
prompt = "The future of AI is"
generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8)
print(generated)
```

### Modernizing the Architecture

**Key Improvements:**

1. **Layer Normalization Placement (Pre-Norm vs Post-Norm)**
```python
# Pre-norm (modern, more stable)
class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Pre-norm: normalize before attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

2. **GELU Activation (vs ReLU)**
```python
# GELU is smoother and often better for transformers
activation = nn.GELU()  # Gaussian Error Linear Unit
```

3. **RMSNorm (Root Mean Square Layer Normalization)**
```python
class RMSNorm(nn.Module):
    """Simpler and faster than LayerNorm"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)
```

4. **SwiGLU Activation**
```python
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
    
    def forward(self, x):
        x, gate = self.linear(x).chunk(2, dim=-1)
        return F.silu(x) * gate  # silu = swish
```

---

## Scaling LLMs for Production

### Scaling Strategies

**1. Model Parallelism**

```python
# Split model across multiple GPUs
class ModelParallelLLM(nn.Module):
    def __init__(self, vocab_size, num_gpus=4):
        super().__init__()
        self.num_gpus = num_gpus
        
        # Split layers across GPUs
        layers_per_gpu = num_layers // num_gpus
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TransformerLayer(...) for _ in range(layers_per_gpu)
            ]).to(f'cuda:{i}') for i in range(num_gpus)
        ])
    
    def forward(self, x):
        for gpu_layers in self.layers:
            for layer in gpu_layers:
                x = layer(x.to(layer.weight.device))
        return x
```

**2. Gradient Checkpointing (Memory Efficiency)**

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientLLM(nn.Module):
    def forward(self, x):
        # Checkpoint intermediate activations
        # Trade compute for memory
        for layer in self.layers:
            x = checkpoint(layer, x)
        return x
```

**3. Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with autocast():
        logits = model(input_ids)
        loss = criterion(logits, labels)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**4. Data Parallelism**

```python
# Use multiple GPUs for data parallelism
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def train_distributed(model, train_loader):
    """Train with distributed data parallel"""
    # Wrap model
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    
    # Training loop
    for batch in train_loader:
        # Forward and backward pass
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Inference Optimization

**1. Quantization**

Quantization reduces model size and speeds up inference by using lower precision (e.g., int8 instead of float32).

**PyTorch Quantization:**

```python
# Dynamic quantization (easiest, good for RNNs/LSTMs)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (better performance, for CNNs)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with sample data
for data in calibration_dataset:
    model(data)
torch.quantization.convert(model, inplace=True)

# Quantization-Aware Training (QAT) - best accuracy
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
# Train normally, quantization is simulated
train_model(model)
torch.quantization.convert(model, inplace=True)
```

**Benefits:**
- 4x smaller model size (float32 → int8)
- 2-4x faster inference
- Lower memory usage
- Better for mobile/edge deployment

**2. Model Pruning**

```python
import torch.nn.utils.prune as prune

# Prune 20% of weights
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)
        prune.remove(module, 'weight')  # Make permanent
```

**3. ONNX Export and Optimization**

**ONNX (Open Neural Network Exchange)** is a format for exporting models to run on different platforms.

```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randint(0, vocab_size, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size'}},
    opset_version=11
)

# Optimize ONNX model
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load and optimize
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Quantize ONNX model
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

**ONNX Runtime (Cross-Platform Inference):**

```python
import onnxruntime as ort

# Load optimized ONNX model
session = ort.InferenceSession("model_quantized.onnx")

# Faster inference than PyTorch
inputs = {"input_ids": input_data.numpy()}
outputs = session.run(None, inputs)

# Benefits:
# - Runs on CPU, GPU, mobile
# - Optimized inference engine
# - 2-5x faster than PyTorch
```

**4. TensorRT (NVIDIA GPU Optimization)**

**TensorRT** optimizes models for NVIDIA GPUs with extreme performance gains.

```python
# First export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Then convert to TensorRT (requires TensorRT SDK)
import tensorrt as trt

# Build TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as model_file:
    parser.parse(model_file.read())

# Configure builder
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # Use FP16 for speed

# Build engine
engine = builder.build_engine(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine.serialize())

# Benefits:
# - 5-10x faster inference on NVIDIA GPUs
# - Automatic kernel fusion
# - Mixed precision (FP16/INT8)
```

**5. GGML/GGUF (LLM Quantization)**

**GGML/GGUF** formats are optimized for running large language models efficiently on CPU and edge devices.

```python
# For LLMs like Llama, Mistral, etc.
# Quantization levels: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

# Using llama.cpp (command line)
# Convert model to GGUF format
# python convert.py model.pt --outtype f16  # FP16
# python convert.py model.pt --outtype q4_0  # 4-bit quantization

# Using Python (if available)
from llama_cpp import Llama

# Load quantized model
llm = Llama(
    model_path="llama-7b-q4_0.gguf",
    n_ctx=2048,
    n_threads=4
)

# Inference
output = llm("Your prompt here", max_tokens=100)

# Benefits:
# - 4-8x smaller model size
# - Runs on CPU efficiently
# - Fast inference on edge devices
# - Used by Ollama, llama.cpp
```

**6. Edge Deployment Formats**

**TensorFlow Lite (Mobile/Edge):**
```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Benefits: Runs on Android, iOS, Raspberry Pi
```

**CoreML (iOS):**
```python
import coremltools as ct

# Convert PyTorch to CoreML
model = ct.convert(
    torch_model,
    inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))]
)

# Quantize
quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
    model, nbits=8
)

model.save("model.mlmodel")
# Benefits: Native iOS deployment, optimized for Apple Silicon
```

**Quantization Comparison:**

| Method | Speedup | Size Reduction | Platform | Best For |
|--------|---------|----------------|----------|----------|
| PyTorch Dynamic | 2x | 4x | CPU | RNNs, quick deployment |
| PyTorch Static | 3-4x | 4x | CPU | CNNs, production |
| ONNX Runtime | 2-5x | 4x | CPU/GPU | Cross-platform |
| TensorRT | 5-10x | 4x | NVIDIA GPU | Production GPU inference |
| GGML/GGUF | 2-4x | 4-8x | CPU/Edge | LLMs, edge devices |
| TFLite | 2-3x | 4x | Mobile | Android/iOS apps |

---

## Mixture-of-Experts (MoE) Architecture

### What is MoE?

**Mixture-of-Experts (MoE)** is an architecture that uses multiple "expert" networks, with a routing mechanism to select which experts process each input. This allows scaling model capacity without proportional increase in computation.

**Key Benefits:**
- **Scalability**: Add more experts without linear compute increase
- **Efficiency**: Only activate subset of experts per input
- **Specialization**: Experts can specialize in different patterns

### MoE Architecture

```python
class Expert(nn.Module):
    """Individual expert network"""
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ffn(x)

class TopKRouter(nn.Module):
    """Router that selects top-k experts"""
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # Calculate routing scores
        logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Softmax over top-k
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        return top_k_probs, top_k_indices

class MoELayer(nn.Module):
    """Mixture-of-Experts layer"""
    def __init__(self, d_model, num_experts=8, top_k=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, dim_feedforward, dropout) 
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = TopKRouter(d_model, num_experts, top_k)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        probs, indices = self.router(x)  # [batch_size, seq_len, top_k]
        
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        probs_flat = probs.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        indices_flat = indices.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        
        # Process through selected experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (indices_flat == expert_idx).any(dim=1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by routing probability
                expert_probs = probs_flat[expert_mask]
                expert_weights = expert_probs[:, (indices_flat[expert_mask] == expert_idx).nonzero(as_tuple=True)[1]]
                
                # Accumulate weighted output
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, d_model)

# Use MoE in transformer
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.moe = MoELayer(d_model, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.moe(self.norm2(x))
        return x
```

### Load Balancing

**Important**: MoE requires load balancing to ensure experts are used evenly.

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    """Calculate load balancing loss to encourage uniform expert usage"""
    # Calculate fraction of tokens routed to each expert
    expert_usage = torch.zeros(num_experts, device=router_probs.device)
    
    for i in range(num_experts):
        mask = (expert_indices == i).any(dim=-1)
        expert_usage[i] = mask.float().mean()
    
    # Load balancing loss (encourage uniform distribution)
    uniform_dist = torch.ones(num_experts, device=router_probs.device) / num_experts
    lb_loss = torch.nn.functional.mse_loss(expert_usage, uniform_dist)
    
    return lb_loss

# Add to training loss
total_loss = model_loss + 0.01 * load_balancing_loss(probs, indices, num_experts)
```

---

## Supervised Fine-Tuning (SFT)

### What is SFT?

**Supervised Fine-Tuning (SFT)** is the process of training a pre-trained language model on a dataset of input-output pairs to adapt it to a specific task or improve its behavior.

**Why SFT?**
- **Task Adaptation**: Adapt general model to specific tasks
- **Style Alignment**: Match desired output style
- **Safety**: Improve model safety and helpfulness
- **Domain Expertise**: Specialize in specific domains

### Single-Task vs Multi-Task Instruction Fine-Tuning

**Single-Task Fine-Tuning:**
Fine-tune on one specific task (e.g., summarization, Q&A, classification).

```python
# Single-task dataset
single_task_data = [
    {"instruction": "Summarize this text", "input": "Long text...", "output": "Summary..."}
]
```

**Multi-Task Instruction Fine-Tuning:**
Train on diverse tasks simultaneously to create a generalist model that can follow instructions across many tasks.

**Benefits:**
- **Generalization**: Better performance on unseen tasks
- **Instruction Following**: Learns to follow diverse instructions
- **Few-Shot Learning**: Better at in-context learning
- **Efficiency**: One model for many tasks

**Implementation:**
```python
# Multi-task dataset format
multi_task_data = [
    # Task 1: Summarization
    {"instruction": "Summarize the following text", "input": "Article...", "output": "Summary"},
    # Task 2: Question Answering
    {"instruction": "Answer the question", "input": "Q: What is ML? A:", "output": "Machine learning is..."},
    # Task 3: Classification
    {"instruction": "Classify the sentiment", "input": "I love this product", "output": "Positive"},
    # Task 4: Code Generation
    {"instruction": "Write a function to", "input": "calculate factorial", "output": "def factorial(n):..."},
    # Task 5: Translation
    {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
]

# Format for training
def format_instruction(example):
    if example["input"]:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

formatted_data = multi_task_data.map(format_instruction)

# Train on diverse tasks
trainer = Trainer(
    model=model,
    train_dataset=formatted_data,
    args=training_args
)
trainer.train()
```

**Best Practices:**
- **Task Diversity**: Include 10-100+ different task types
- **Data Balance**: Balance examples across tasks
- **Instruction Quality**: Clear, consistent instruction format
- **Evaluation**: Test on held-out tasks to measure generalization

### SFT Dataset Format

```python
# SFT data format: instruction-response pairs
sft_dataset = [
    {
        "instruction": "Explain machine learning in simple terms",
        "response": "Machine learning is a way for computers to learn from data..."
    },
    {
        "instruction": "Write a Python function to calculate factorial",
        "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    }
]
```

### SFT Implementation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

class SFTDataset:
    """Dataset for supervised fine-tuning"""
    
    def __init__(self, tokenizer, instructions, responses, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Format: "### Instruction:\n{instruction}\n\n### Response:\n{response}"
        self.examples = []
        for inst, resp in zip(instructions, responses):
            text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
            self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (only compute loss on response part)
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # Mask instruction part (don't compute loss on it)
        instruction_end = text.find("### Response:")
        instruction_tokens = self.tokenizer(
            text[:instruction_end],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze()
        
        labels[:len(instruction_tokens)] = -100  # Ignore in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

def train_sft(model_name, train_data, output_dir='./sft_model'):
    """Train supervised fine-tuning"""
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Prepare dataset
    instructions = [item['instruction'] for item in train_data]
    responses = [item['response'] for item in train_data]
    dataset = SFTDataset(tokenizer, instructions, responses)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        fp16=True,  # Mixed precision
        remove_unused_columns=False,
    )
    
    # Custom data collator
    def data_collator(features):
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
        return batch
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    
    return model, tokenizer

# Example usage
train_data = [
    {"instruction": "What is Python?", "response": "Python is a high-level programming language..."},
    # ... more examples
]

model, tokenizer = train_sft("gpt2", train_data)
```

### Parameter Efficient Fine-Tuning (PEFT)

**Why PEFT?**
Full fine-tuning requires updating all model parameters, which is:
- **Memory Intensive**: Need to store gradients for all parameters
- **Slow**: Many parameters to update
- **Expensive**: High compute costs
- **Storage**: Need full model copy per task

**PEFT Techniques:**
Update only a small subset of parameters while keeping the base model frozen.

#### 1. LoRA (Low-Rank Adaptation)

**Concept**: Decompose weight updates into low-rank matrices.

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank (low-rank dimension)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Which layers to apply LoRA
)

# Apply LoRA to model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(model, lora_config)

# Now only LoRA parameters are trainable
print(f"Trainable parameters: {model.num_parameters(trainable=True):,}")
print(f"Total parameters: {model.num_parameters():,}")

# Train normally
trainer = Trainer(model=model, ...)
trainer.train()

# Save only LoRA weights (much smaller)
model.save_pretrained("./lora_model")  # Saves only ~10MB instead of GBs
```

**Benefits:**
- **Memory**: 10-100x less memory than full fine-tuning
- **Speed**: Faster training (fewer parameters)
- **Storage**: Only save adapter weights (~10MB vs GBs)
- **Modularity**: Multiple LoRA adapters for different tasks

**LoRA Mathematics:**
Original weight update: `W' = W + ΔW`
LoRA approximation: `ΔW ≈ BA` where `B ∈ R^(d×r)`, `A ∈ R^(r×k)`, `r << min(d,k)`

#### 2. Soft Prompts (Prompt Tuning)

**Concept**: Learn continuous prompt embeddings instead of discrete tokens.

```python
from peft import PromptTuningConfig, get_peft_model

# Configure soft prompts
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # Number of learnable prompt tokens
    prompt_tuning_init="TEXT",  # Initialize from text
    prompt_tuning_init_text="Classify the sentiment of this text:",
)

# Apply to model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(model, prompt_config)

# Train - only prompt embeddings are updated
trainer = Trainer(model=model, ...)
trainer.train()
```

**Benefits:**
- **Extremely Efficient**: Only train prompt embeddings
- **Task-Specific**: Different prompts for different tasks
- **No Model Changes**: Keep base model completely frozen

**Comparison:**

| Method | Trainable Params | Memory | Speed | Best For |
|--------|------------------|--------|-------|----------|
| Full Fine-tuning | 100% | High | Slow | Maximum performance |
| LoRA | 0.1-1% | Low | Fast | General purpose |
| Soft Prompts | <0.01% | Very Low | Very Fast | Simple tasks |

### SFT Best Practices

**1. Data Quality**
- High-quality, diverse examples
- Consistent formatting
- Appropriate length (not too short/long)

**2. Learning Rate**
- Lower learning rate than pre-training (1e-5 to 5e-5)
- Use learning rate scheduling

**3. Training Duration**
- Usually 1-3 epochs (avoid overfitting)
- Monitor validation loss

**4. PEFT Selection**
- **LoRA**: Best balance of performance and efficiency
- **Soft Prompts**: For simple tasks, maximum efficiency
- **Full Fine-tuning**: When maximum performance is needed

**4. Prompt Formatting**
- Consistent instruction format
- Clear separation between instruction and response

---

## Model Evaluation and Benchmarks

### Why Evaluation Matters

Evaluating LLMs is crucial for:
- **Performance Assessment**: Measure model capabilities
- **Comparison**: Compare different models and approaches
- **Progress Tracking**: Monitor improvements during training
- **Deployment Decisions**: Determine if model is production-ready

### Evaluation Metrics

**1. Perplexity**
- Measures how well model predicts test data
- Lower is better
- Formula: `PPL = exp(cross_entropy_loss)`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for a text"""
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
```

**2. Task-Specific Metrics**
- **Classification**: Accuracy, F1-score, Precision, Recall
- **Generation**: BLEU, ROUGE, METEOR
- **Question Answering**: Exact Match, F1-score
- **Summarization**: ROUGE-L, BLEU

**3. Human Evaluation**
- **Quality**: Fluency, coherence, relevance
- **Safety**: Harmful content, bias
- **Helpfulness**: Task completion, correctness

### Standard Benchmarks

**1. GLUE (General Language Understanding Evaluation)**
- 9 tasks: sentiment, paraphrase, inference, etc.
- Measures general language understanding

**2. SuperGLUE**
- More challenging version of GLUE
- Includes reading comprehension, reasoning

**3. MMLU (Massive Multitask Language Understanding)**
- 57 tasks across STEM, humanities, social sciences
- Measures broad knowledge and reasoning

**4. HELM (Holistic Evaluation of Language Models)**
- Comprehensive evaluation across many scenarios
- Includes accuracy, calibration, robustness, fairness

**5. BIG-bench (Beyond the Imitation Game)**
- 200+ diverse tasks
- Tests reasoning, knowledge, creativity

**6. HumanEval**
- Code generation benchmark
- Measures functional correctness

**7. TruthfulQA**
- Measures truthfulness and avoiding falsehoods
- Important for safety evaluation

### Evaluation Best Practices

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load benchmark
mmlu = load_dataset("cais/mmlu", "all", split="test")

# Evaluate model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def evaluate_mmlu(model, tokenizer, dataset, subject="all"):
    """Evaluate on MMLU benchmark"""
    correct = 0
    total = 0
    
    for example in dataset:
        if subject != "all" and example["subject"] != subject:
            continue
        
        # Format question
        prompt = f"{example['question']}\nA. {example['choices'][0]}\nB. {example['choices'][1]}\nC. {example['choices'][2]}\nD. {example['choices'][3]}\nAnswer:"
        
        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=1)
        answer = tokenizer.decode(outputs[0][-1])
        
        # Check correctness
        correct_answer = example['choices'][example['answer']]
        if answer.strip().upper() == correct_answer[0]:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Evaluate
accuracy = evaluate_mmlu(model, tokenizer, mmlu, subject="all")
print(f"MMLU Accuracy: {accuracy:.2%}")
```

**Key Principles:**
- **Multiple Metrics**: Don't rely on a single metric
- **Diverse Tasks**: Evaluate across different capabilities
- **Human Evaluation**: Complement automated metrics
- **Robustness**: Test on out-of-distribution data
- **Bias Testing**: Evaluate fairness across groups

---

## Reward Modeling for RLHF

### What is Reward Modeling?

**Reward Modeling** is the process of training a model to predict human preferences, creating a reward signal for reinforcement learning.

**Purpose:**
- **Preference Learning**: Learn what humans prefer
- **Alignment**: Align model outputs with human values
- **Safety**: Reward safe and helpful responses

### Reward Model Architecture

```python
class RewardModel(nn.Module):
    """Reward model for RLHF"""
    
    def __init__(self, base_model_name, hidden_size=768):
        super().__init__()
        # Load base language model
        from transformers import AutoModel
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Reward head (maps hidden states to scalar reward)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output'):
            hidden = outputs.pooler_output
        else:
            # Mean pooling
            hidden = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        # Predict reward
        reward = self.reward_head(hidden)
        return reward.squeeze(-1)
```

### Training Reward Model

```python
def train_reward_model(reward_model, preference_data, num_epochs=3):
    """Train reward model on preference data"""
    
    optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in preference_data:
            # Each batch contains: chosen_text, rejected_text
            chosen_ids = batch['chosen_input_ids']
            rejected_ids = batch['rejected_input_ids']
            chosen_mask = batch['chosen_attention_mask']
            rejected_mask = batch['rejected_attention_mask']
            
            # Get rewards
            chosen_reward = reward_model(chosen_ids, chosen_mask)
            rejected_reward = reward_model(rejected_ids, rejected_mask)
            
            # Loss: chosen should have higher reward
            # Use ranking loss (Bradley-Terry model)
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(preference_data):.4f}")

# Preference data format
preference_data = [
    {
        "chosen": "Helpful and accurate response",
        "rejected": "Unhelpful or incorrect response"
    }
]
```

### Preference Data Collection

```python
# Example: Collecting preferences
def collect_preferences(prompts, model_responses):
    """Collect human preferences for model responses"""
    preferences = []
    
    for prompt, responses in zip(prompts, model_responses):
        # responses is a list of candidate responses
        # Human annotator ranks them
        ranked = human_rank(responses)  # Returns indices in preference order
        
        # Create pairs: better > worse
        for i in range(len(ranked) - 1):
            for j in range(i + 1, len(ranked)):
                preferences.append({
                    "prompt": prompt,
                    "chosen": responses[ranked[i]],
                    "rejected": responses[ranked[j]]
                })
    
    return preferences
```

---

## Reinforcement Learning from Human Feedback (RLHF)

### What is RLHF?

**RLHF (Reinforcement Learning from Human Feedback)** is a training method that uses reinforcement learning to optimize language models based on human preferences, typically using PPO (Proximal Policy Optimization).

**RLHF Pipeline:**
1. **Pre-train**: Train base language model
2. **SFT**: Supervised fine-tuning on instruction-following data
3. **Reward Modeling**: Train reward model on human preferences
4. **RLHF**: Optimize policy (model) using PPO with reward model

### PPO for RLHF

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class PPOAgent:
    """PPO agent for RLHF"""
    
    def __init__(self, model_name, reward_model, tokenizer):
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)  # Frozen reference
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def generate(self, prompts, max_length=128, temperature=1.0):
        """Generate responses using current policy"""
        self.policy_model.eval()
        
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                
                outputs = self.policy_model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                responses.append(response)
        
        return responses
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards using reward model"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}{response}"
            input_ids = self.tokenizer.encode(full_text, return_tensors='pt')
            
            with torch.no_grad():
                reward = self.reward_model(input_ids)
                rewards.append(reward.item())
        
        return torch.tensor(rewards)
    
    def ppo_update(self, prompts, responses, old_log_probs, advantages, clip_epsilon=0.2):
        """Perform PPO update"""
        self.policy_model.train()
        
        # Compute new log probabilities
        new_log_probs = []
        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}{response}"
            input_ids = self.tokenizer.encode(full_text, return_tensors='pt')
            
            outputs = self.policy_model(input_ids, labels=input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            
            # Get log prob of generated sequence
            response_ids = self.tokenizer.encode(response, return_tensors='pt')
            response_log_prob = log_probs[0, :len(response_ids[0]), response_ids[0]].mean()
            new_log_probs.append(response_log_prob)
        
        new_log_probs = torch.stack(new_log_probs)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL penalty (prevent policy from drifting too far)
        with torch.no_grad():
            ref_log_probs = []
            for prompt, response in zip(prompts, responses):
                full_text = f"{prompt}{response}"
                input_ids = self.tokenizer.encode(full_text, return_tensors='pt')
                outputs = self.ref_model(input_ids, labels=input_ids)
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                response_ids = self.tokenizer.encode(response, return_tensors='pt')
                ref_log_prob = log_probs[0, :len(response_ids[0]), response_ids[0]].mean()
                ref_log_probs.append(ref_log_prob)
        
        ref_log_probs = torch.stack(ref_log_probs)
        kl_penalty = (new_log_probs - ref_log_probs).mean()
        
        # Total loss
        total_loss = policy_loss + 0.1 * kl_penalty
        
        return total_loss

def train_rlhf(agent, prompts, num_iterations=100):
    """Train using RLHF"""
    optimizer = optim.AdamW(agent.policy_model.parameters(), lr=1e-6)
    
    for iteration in range(num_iterations):
        # Generate responses
        responses = agent.generate(prompts)
        
        # Compute rewards
        rewards = agent.compute_rewards(prompts, responses)
        
        # Compute advantages (normalize rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get old log probs (from previous iteration)
        old_log_probs = agent.get_log_probs(prompts, responses)
        
        # PPO update
        for _ in range(4):  # Multiple PPO epochs
            loss = agent.ppo_update(prompts, responses, old_log_probs, advantages)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_model.parameters(), 1.0)
            optimizer.step()
        
        # Update reference model periodically
        if iteration % 10 == 0:
            agent.ref_model.load_state_dict(agent.policy_model.state_dict())
        
        print(f"Iteration {iteration+1}, Reward: {rewards.mean():.4f}")
```

### RLHF Best Practices

**1. Reward Model Quality**
- High-quality preference data
- Diverse prompts and responses
- Regular reward model updates

**2. PPO Hyperparameters**
- Small learning rate (1e-6 to 1e-5)
- Clip epsilon: 0.1 to 0.3
- KL penalty: 0.01 to 0.1

**3. Training Stability**
- Update reference model periodically
- Monitor KL divergence
- Use gradient clipping

**4. Evaluation**
- Human evaluation on held-out set
- Measure helpfulness, harmlessness, honesty
- Track reward model accuracy

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

