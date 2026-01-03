# Graph Neural Networks Advanced Topics

Advanced topics in Graph Neural Networks for research and production applications.

## Table of Contents

- [Graph Transformers](#graph-transformers)
- [Dynamic Graphs and Temporal GNNs](#dynamic-graphs-and-temporal-gnns)
- [Heterogeneous Graphs](#heterogeneous-graphs)
- [Graph Pooling Strategies](#graph-pooling-strategies)
- [Graph Generation](#graph-generation)
- [Scalable GNNs](#scalable-gnns)
- [Graph Contrastive Learning](#graph-contrastive-learning)
- [Resources and Further Reading](#resources-and-further-reading)

---

## Graph Transformers

### Overview

**Graph Transformers** apply the transformer architecture to graphs, using self-attention over nodes while incorporating graph structure.

### Key Differences from Standard Transformers

1. **Positional Encoding**: Instead of sequence position, encode graph structure (e.g., Laplacian eigenvectors, random walk features)
2. **Graph-aware Attention**: Mask attention based on graph connectivity
3. **Edge Features**: Incorporate edge information into attention mechanism

### Implementation

```python
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_adj

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=3):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Linear(10, hidden_dim)  # Positional encoding
        
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)
    
    def forward(self, x, edge_index, pos_enc=None):
        # Node embedding
        x = self.embedding(x)
        
        # Add positional encoding if provided
        if pos_enc is not None:
            x = x + self.pos_encoder(pos_enc)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Output
        x = self.fc(x)
        return x
```

### Positional Encoding for Graphs

```python
import numpy as np
from scipy.sparse.linalg import eigs

def laplacian_positional_encoding(adj, k=10):
    """
    Compute Laplacian eigenvectors as positional encoding
    """
    # Compute normalized Laplacian
    degree = adj.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    L = np.eye(adj.shape[0]) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Get k smallest eigenvectors
    eigenvalues, eigenvectors = eigs(L, k=k, which='SM')
    return eigenvectors.real
```

### Advantages

- **Long-range Dependencies**: Can attend to distant nodes
- **Flexible Architecture**: No fixed graph structure required
- **Parallelizable**: Self-attention is parallelizable

### Limitations

- **Quadratic Complexity**: O(N²) attention over all nodes
- **Memory Intensive**: Requires full adjacency matrix
- **Less Graph-aware**: May not fully utilize graph structure

---

## Dynamic Graphs and Temporal GNNs

### Temporal Graphs

**Temporal Graphs** change over time with nodes/edges being added or removed.

**Applications:**
- Social networks evolving over time
- Traffic networks with changing flow
- Financial transaction networks
- Citation networks growing over time

### Temporal Graph Network (TGN)

**TGN** handles time-evolving graphs by maintaining memory for each node.

```python
from torch_geometric.nn import TGNMemory, TransformerConv

class TemporalGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, memory_dim, time_dim):
        super(TemporalGNN, self).__init__()
        self.memory = TGNMemory(
            node_dim, 
            memory_dim, 
            time_dim,
            message_module=nn.Linear(node_dim + edge_dim + time_dim, memory_dim),
            aggregator_module=None
        )
        self.gnn = TransformerConv(memory_dim, memory_dim)
    
    def forward(self, src, dst, t, msg, edge_attr):
        # Update memory
        self.memory.update_state(src, dst, t, msg)
        
        # Get memory embeddings
        memory = self.memory.get_memory()
        
        # GNN on memory
        out = self.gnn(memory, edge_index)
        return out
```

### Dynamic Graph Convolution

```python
class DynamicGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DynamicGCN, self).__init__()
        self.conv = GCNConv(input_dim, hidden_dim)
        self.temporal_conv = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x_list, edge_index_list):
        # x_list: list of node features at different time steps
        # edge_index_list: list of edge indices at different time steps
        
        hidden_states = []
        for x, edge_index in zip(x_list, edge_index_list):
            h = self.conv(x, edge_index)
            hidden_states.append(h)
        
        # Temporal modeling
        hidden_tensor = torch.stack(hidden_states, dim=1)  # [N, T, H]
        out, _ = self.temporal_conv(hidden_tensor)
        out = self.fc(out[:, -1])  # Use last time step
        return out
```

---

## Heterogeneous Graphs

### Concept

**Heterogeneous Graphs** have multiple node types and edge types, representing complex relationships.

**Example**: Academic Network
- **Node Types**: Papers, Authors, Venues, Topics
- **Edge Types**: Cites, Writes, Publishes, HasTopic

### Relational Graph Convolutional Network (RGCN)

**RGCN** handles multiple edge types by using separate weight matrices for each relation.

```python
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)
    
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x
```

### Heterogeneous Graph Attention Network (HAN)

**HAN** uses attention for both node-level and semantic-level aggregation.

```python
from torch_geometric.nn import HANConv

class HAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_metapaths):
        super(HAN, self).__init__()
        self.conv1 = HANConv(input_dim, hidden_dim, num_metapaths)
        self.conv2 = HANConv(hidden_dim, output_dim, num_metapaths)
    
    def forward(self, x, edge_index_dict):
        x = self.conv1(x, edge_index_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict)
        return x
```

---

## Graph Pooling Strategies

### Graph-level Representations

**Graph Pooling** aggregates node representations to obtain graph-level embeddings for graph classification.

### Global Pooling Methods

```python
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# Mean pooling
graph_embedding = global_mean_pool(node_embeddings, batch)

# Max pooling
graph_embedding = global_max_pool(node_embeddings, batch)

# Sum pooling
graph_embedding = global_add_pool(node_embeddings, batch)
```

### Attention-based Pooling

```python
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention weights
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Weighted sum
        graph_embedding = (x * attn_weights).sum(dim=0)
        return graph_embedding
```

### Hierarchical Pooling (DiffPool)

**DiffPool** learns to cluster nodes hierarchically.

```python
from torch_geometric.nn import DiffPool

class HierarchicalPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        super(HierarchicalPooling, self).__init__()
        self.pool = DiffPool(input_dim, hidden_dim, num_clusters)
    
    def forward(self, x, edge_index, batch):
        x, edge_index, batch = self.pool(x, edge_index, batch)
        return x, edge_index, batch
```

---

## Graph Generation

### Generating Graphs

**Graph Generation** creates new graph structures, useful for molecule generation, social network synthesis, etc.

### Variational Graph Autoencoder (VGAE)

**VGAE** learns to generate graphs by encoding and decoding graph structure.

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * input_dim)
        )
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        adj_logits = self.decoder(z)
        adj_logits = adj_logits.view(-1, adj_logits.size(-1))
        return torch.sigmoid(adj_logits)
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decode(z)
        return adj_recon, mu, logvar
```

### Graph GANs

**Graph GANs** use adversarial training to generate realistic graphs.

```python
class GraphGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super(GraphGenerator, self).__init__()
        self.max_nodes = max_nodes
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, max_nodes * max_nodes)
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        adj_logits = self.fc2(x)
        adj_logits = adj_logits.view(-1, self.max_nodes, self.max_nodes)
        adj = torch.sigmoid(adj_logits)
        return adj

class GraphDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphDiscriminator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch=None)
        x = self.fc(x)
        return torch.sigmoid(x)
```

---

## Scalable GNNs

### Challenges with Large Graphs

- **Memory**: Full graph doesn't fit in memory
- **Computation**: O(N²) operations for large N
- **Training Time**: Slow convergence

### Solutions

#### 1. Neighborhood Sampling

```python
from torch_geometric.loader import NeighborSampler

# Sample neighbors for each node
sampler = NeighborSampler(
    edge_index,
    sizes=[25, 10],  # Sample 25 neighbors at layer 1, 10 at layer 2
    batch_size=1024,
    num_workers=4
)

for batch_size, n_id, adjs in sampler:
    # adjs: List of (edge_index, e_id, size) tuples
    # Process sampled subgraph
    pass
```

#### 2. Cluster-based Sampling

```python
from torch_geometric.loader import ClusterLoader
from torch_geometric.utils import to_undirected

# Partition graph into clusters
cluster_loader = ClusterLoader(
    data,
    num_parts=10,  # Partition into 10 clusters
    batch_size=1
)

for subgraph in cluster_loader:
    # Process each cluster
    pass
```

#### 3. GraphSAINT

**GraphSAINT** samples subgraphs for training.

```python
from torch_geometric.loader import GraphSAINTSampler

sampler = GraphSAINTSampler(
    data,
    batch_size=6000,
    num_steps=5,
    sample_coverage=100
)
```

---

## Graph Contrastive Learning

### Self-Supervised Learning on Graphs

**Graph Contrastive Learning** learns representations without labels by contrasting positive and negative pairs.

### GraphCL

**GraphCL** uses data augmentation and contrastive learning.

```python
class GraphCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim):
        super(GraphCL, self).__init__()
        self.encoder = GCN(input_dim, hidden_dim, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        z = self.projection(h)
        return F.normalize(z, dim=1)
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        # z1, z2: augmented views of same graph
        batch_size = z1.size(0)
        
        # Positive pairs: same graph
        pos_sim = F.cosine_similarity(z1, z2, dim=1)
        
        # Negative pairs: different graphs
        neg_sim = torch.matmul(z1, z2.t()) / temperature
        
        # Contrastive loss
        labels = torch.arange(batch_size).to(z1.device)
        loss = F.cross_entropy(neg_sim, labels)
        return loss
```

---

## Resources and Further Reading

### Important Papers

1. **"Semi-Supervised Classification with Graph Convolutional Networks"** - Kipf & Welling, 2017 (GCN)
2. **"Graph Attention Networks"** - Veličković et al., 2018 (GAT)
3. **"Inductive Representation Learning on Large Graphs"** - Hamilton et al., 2017 (GraphSAGE)
4. **"How Powerful are Graph Neural Networks?"** - Xu et al., 2019 (GIN)
5. **"Do Transformers Really Perform Bad for Graph Representation?"** - Ying et al., 2021 (Graph Transformer)
6. **"Temporal Graph Networks for Deep Learning on Dynamic Graphs"** - Rossi et al., 2020 (TGN)
7. **"Modeling Relational Data with Graph Convolutional Networks"** - Schlichtkrull et al., 2018 (RGCN)

### Datasets

1. **Open Graph Benchmark (OGB)**: Large-scale graph datasets
   - [OGB Website](https://ogb.stanford.edu/)
   - Node, link, and graph prediction tasks

2. **TUDataset**: Collection of graph datasets
   - Molecular graphs, social networks

3. **Cora, CiteSeer, PubMed**: Citation networks

### Tools

1. **PyTorch Geometric**: Most popular GNN library
2. **Deep Graph Library (DGL)**: Efficient and scalable
3. **StellarGraph**: Comprehensive graph ML library
4. **Spektral**: Keras/TensorFlow GNN library

### Courses and Tutorials

1. **CS224W: Machine Learning with Graphs** - Stanford
2. **PyTorch Geometric Tutorials**: Comprehensive examples
3. **DGL Tutorials**: Step-by-step guides

---

## Key Takeaways

1. **Graph Transformers**: Powerful for long-range dependencies but computationally expensive
2. **Dynamic Graphs**: Require temporal modeling (TGN, dynamic GCN)
3. **Heterogeneous Graphs**: Need specialized architectures (RGCN, HAN)
4. **Graph Pooling**: Essential for graph-level tasks (mean, max, attention, hierarchical)
5. **Graph Generation**: Useful for molecule design, network synthesis (VGAE, GANs)
6. **Scalability**: Use sampling, clustering, or efficient architectures for large graphs
7. **Self-Supervised Learning**: Contrastive learning enables learning without labels
8. **Choose Architecture**: Based on graph type, size, and task requirements

