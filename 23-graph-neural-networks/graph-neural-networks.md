# Graph Neural Networks Complete Guide

Comprehensive guide to Graph Neural Networks (GNNs) for learning on graph-structured data.

## Table of Contents

- [Introduction to Graphs and GNNs](#introduction-to-graphs-and-gnns)
- [Graph Representations](#graph-representations)
- [Message Passing](#message-passing)
- [Graph Convolutional Networks (GCNs)](#graph-convolutional-networks-gcns)
- [Graph Attention Networks (GATs)](#graph-attention-networks-gats)
- [Other GNN Architectures](#other-gnn-architectures)
- [GNN Libraries](#gnn-libraries)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Graphs and GNNs

### What are Graphs?

**Graphs** are data structures consisting of:
- **Nodes (Vertices)**: Entities in the graph
- **Edges**: Relationships between nodes
- **Features**: Node features, edge features, graph-level features

### Why Graphs?

Many real-world problems involve graph-structured data:
- **Social Networks**: Users (nodes) and friendships (edges)
- **Molecules**: Atoms (nodes) and bonds (edges)
- **Knowledge Graphs**: Entities (nodes) and relations (edges)
- **Citation Networks**: Papers (nodes) and citations (edges)
- **Recommendation Systems**: Users and items (nodes), interactions (edges)

### What are Graph Neural Networks?

**Graph Neural Networks (GNNs)** are neural networks designed to operate on graph-structured data. They learn representations of nodes, edges, or entire graphs.

**Key Idea**: Aggregate information from neighbors to learn node representations.

---

## Graph Representations

### Adjacency Matrix

**Adjacency Matrix (A)**: Square matrix where A[i,j] = 1 if edge exists between nodes i and j, else 0.

```python
import numpy as np

# Example: 4-node graph
# 0 -- 1
# |    |
# 2 -- 3

A = np.array([
    [0, 1, 1, 0],  # Node 0 connected to 1, 2
    [1, 0, 0, 1],  # Node 1 connected to 0, 3
    [1, 0, 0, 1],  # Node 2 connected to 0, 3
    [0, 1, 1, 0]   # Node 3 connected to 1, 2
])
```

### Edge List

**Edge List**: List of (source, target) pairs.

```python
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
```

### Node Features

**Node Features (X)**: Feature matrix where each row is a node's feature vector.

```python
# Example: 4 nodes, each with 3 features
X = np.array([
    [1.0, 0.5, 0.2],  # Node 0 features
    [0.8, 0.3, 0.1],  # Node 1 features
    [0.6, 0.4, 0.3],  # Node 2 features
    [0.9, 0.2, 0.4]   # Node 3 features
])
```

---

## Message Passing

### Core Concept

**Message Passing** is the fundamental operation in GNNs:

1. **Message**: Each node sends information to its neighbors
2. **Aggregation**: Each node aggregates messages from neighbors
3. **Update**: Each node updates its representation

### Mathematical Formulation

For node v at layer l:

```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

Where:
- `h_v^(l)`: Representation of node v at layer l
- `N(v)`: Neighbors of node v
- `UPDATE`: Update function
- `AGGREGATE`: Aggregation function (sum, mean, max, etc.)

### Simple Example

```python
import torch
import torch.nn as nn

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, adj):
        # x: node features [N, input_dim]
        # adj: adjacency matrix [N, N]
        
        # Aggregate: sum of neighbor features
        neighbor_sum = torch.matmul(adj, x)  # [N, input_dim]
        
        # Update: linear transformation
        out = self.linear(neighbor_sum)  # [N, output_dim]
        
        return out
```

---

## Graph Convolutional Networks (GCNs)

### GCN Layer

**Graph Convolutional Network (GCN)** performs convolution on graphs.

#### GCN Formula

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

Where:
- `H^(l)`: Node features at layer l
- `A`: Adjacency matrix
- `D`: Degree matrix (diagonal matrix with node degrees)
- `W^(l)`: Learnable weight matrix
- `σ`: Activation function

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # x: [N, in_features]
        # adj: [N, N] (normalized adjacency matrix)
        
        # Linear transformation
        x = self.linear(x)  # [N, out_features]
        
        # Graph convolution
        x = torch.matmul(adj, x)  # [N, out_features]
        
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x
```

### Normalization

**Normalized Adjacency Matrix**:

```python
def normalize_adjacency(adj):
    # Add self-loops
    adj = adj + torch.eye(adj.size(0))
    
    # Compute degree matrix
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    
    # Normalize
    adj_normalized = torch.matmul(
        torch.matmul(degree_matrix_inv_sqrt, adj),
        degree_matrix_inv_sqrt
    )
    
    return adj_normalized
```

---

## Graph Attention Networks (GATs)

### Attention Mechanism

**Graph Attention Network (GAT)** uses attention to learn importance of neighbors.

### GAT Formula

```
h_v^(l+1) = σ(Σ(u∈N(v)) α_vu W^(l) h_u^(l))
```

Where `α_vu` is the attention weight:

```
α_vu = softmax(LeakyReLU(a^T [W h_v || W h_u]))
```

### Implementation

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features * num_heads)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, adj):
        # x: [N, in_features]
        # adj: [N, N]
        
        N = x.size(0)
        Wh = self.W(x)  # [N, out_features * num_heads]
        Wh = Wh.view(N, self.num_heads, self.out_features)
        
        # Compute attention
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Masked attention
        attention = torch.where(adj > 0, e, torch.tensor(-9e15))
        attention = F.softmax(attention, dim=1)
        
        # Apply attention
        h_prime = torch.matmul(attention.unsqueeze(1), Wh).squeeze(1)
        
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        # Prepare for attention computation
        N = Wh.size(0)
        Wh1 = Wh.unsqueeze(1).expand(N, N, self.num_heads, self.out_features)
        Wh2 = Wh.unsqueeze(0).expand(N, N, self.num_heads, self.out_features)
        return torch.cat([Wh1, Wh2], dim=-1)
```

---

## Other GNN Architectures

### GraphSAGE

**GraphSAGE (Graph Sample and Aggregate)** learns node embeddings by sampling and aggregating from neighborhoods.

**Key Features:**
- Inductive learning (works on new graphs)
- Neighborhood sampling
- Aggregation functions (mean, max, LSTM)

### Graph Isomorphism Network (GIN)

**GIN** is provably as powerful as the Weisfeiler-Lehman test.

### Graph Transformer

**Graph Transformer** applies transformer architecture to graphs.

---

## GNN Libraries

### PyTorch Geometric

```python
import torch_geometric
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

### Deep Graph Library (DGL)

```python
import dgl
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats)
        self.conv2 = dglnn.GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

---

## Practice Exercises

1. **Node Classification**: Classify nodes in Cora citation network
2. **Link Prediction**: Predict missing edges
3. **Graph Classification**: Classify molecular graphs
4. **Recommendation System**: Build GNN-based recommender
5. **Knowledge Graph**: Learn entity embeddings

---

## Key Takeaways

1. **Graphs are Everywhere**: Many problems have graph structure
2. **Message Passing**: Core mechanism of GNNs
3. **Neighborhood Aggregation**: Combine information from neighbors
4. **Attention**: Learn importance of neighbors (GAT)
5. **Libraries**: Use PyTorch Geometric or DGL for efficiency
6. **Applications**: Social networks, molecules, knowledge graphs

---

**Next Steps**: Explore [Advanced Topics](graph-neural-networks-advanced-topics.md) for graph transformers, dynamic graphs, and heterogeneous graphs.

