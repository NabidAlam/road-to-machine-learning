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

**GraphSAGE (Graph Sample and Aggregate)** learns node embeddings by sampling and aggregating from neighborhoods. Unlike GCN which requires the full graph, GraphSAGE can work inductively on new nodes.

#### Key Features

1. **Inductive Learning**: Works on new graphs and nodes not seen during training
2. **Neighborhood Sampling**: Samples fixed-size neighborhoods for efficiency
3. **Aggregation Functions**: Mean, max, or LSTM aggregators

#### GraphSAGE Algorithm

```
For each node v:
  1. Sample k neighbors uniformly
  2. Aggregate neighbor features: h_N(v) = AGGREGATE({h_u : u ∈ N(v)})
  3. Update: h_v = σ(W · CONCAT(h_v, h_N(v)))
```

#### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Usage
model = GraphSAGE(input_dim=1433, hidden_dim=64, output_dim=7)
```

#### Aggregation Functions

**Mean Aggregator**:
```python
def mean_aggregate(neighbor_features):
    return torch.mean(neighbor_features, dim=0)
```

**Max Aggregator**:
```python
def max_aggregate(neighbor_features):
    return torch.max(neighbor_features, dim=0)[0]
```

**LSTM Aggregator**:
```python
class LSTMAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, neighbor_features):
        # neighbor_features: [num_neighbors, feature_dim]
        out, _ = self.lstm(neighbor_features.unsqueeze(0))
        return out[0, -1]  # Return last hidden state
```

### Graph Isomorphism Network (GIN)

**Graph Isomorphism Network (GIN)** is provably as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test, making it one of the most expressive GNN architectures.

#### Why GIN?

- **Maximum Expressive Power**: Can distinguish any non-isomorphic graphs
- **Theoretical Guarantees**: Based on graph theory
- **Simple Architecture**: Easy to implement

#### GIN Formula

```
h_v^(l+1) = MLP^((l))((1 + ε^(l)) · h_v^(l) + Σ(u∈N(v)) h_u^(l))
```

Where:
- `ε^(l)`: Learnable parameter (can be fixed to 0)
- `MLP^((l))`: Multi-layer perceptron at layer l

#### Implementation

```python
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_hidden, train_eps=True))
        
        # Output layer
        nn_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.convs.append(GINConv(nn_out, train_eps=True))
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x
```

### Graph Transformer

**Graph Transformer** applies the transformer architecture to graphs, using self-attention over nodes.

#### Key Components

1. **Positional Encoding**: Encode graph structure (e.g., Laplacian eigenvectors)
2. **Self-Attention**: Attention over all nodes
3. **Graph-aware Mechanisms**: Incorporate edge information

#### Implementation

```python
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=3):
        super(GraphTransformer, self).__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        self.convs.append(TransformerConv(hidden_dim * num_heads, output_dim, heads=1))
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```

### Gated Graph Neural Networks (GGNN)

**GGNN** uses gated recurrent units (GRUs) for message passing.

```python
from torch_geometric.nn import GatedGraphConv

class GGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GGNN, self).__init__()
        self.conv = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.fc(x)
        return x
```

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

## GNN Tasks and Applications

### Node Classification

**Node Classification** assigns labels to nodes in a graph.

**Example**: Classify research papers in a citation network by subject.

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Model
class NodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training
model = NodeClassifier(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
```

### Link Prediction

**Link Prediction** predicts whether an edge exists between two nodes.

**Example**: Predict friendships in social networks or citations in academic networks.

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

class LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinkPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # Get node embeddings for edge endpoints
        row, col = edge_index
        z_i = z[row]
        z_j = z[col]
        # Concatenate and predict
        return torch.sigmoid(self.fc(torch.cat([z_i, z_j], dim=1)))
    
    def forward(self, x, edge_index, neg_edge_index=None):
        z = self.encode(x, edge_index)
        pos_pred = self.decode(z, edge_index)
        if neg_edge_index is not None:
            neg_pred = self.decode(z, neg_edge_index)
            return pos_pred, neg_pred
        return pos_pred

# Training
model = LinkPredictor(dataset.num_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    
    # Positive edges
    pos_pred = model.decode(z, data.edge_index)
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones(pos_pred.size(0), 1))
    
    # Negative edges
    neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
    neg_pred = model.decode(z, neg_edge_index)
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros(neg_pred.size(0), 1))
    
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Graph Classification

**Graph Classification** assigns labels to entire graphs.

**Example**: Classify molecules by their properties (e.g., toxicity, solubility).

```python
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # Graph-level representation (pooling)
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Load molecular dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
```

### Recommendation Systems

**GNN-based Recommendation** uses graph structure of user-item interactions.

**Example**: Recommend products to users based on interaction graph.

```python
class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(GNNRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, user_idx, item_idx, edge_index):
        # Initialize node features
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Get user and item embeddings
        user_emb = x[user_idx]
        item_emb = x[item_idx + self.user_embedding.num_embeddings]
        
        # Predict rating
        rating = self.fc((user_emb * item_emb).sum(dim=1))
        return torch.sigmoid(rating)
```

## Common Challenges and Solutions

### Over-smoothing

**Problem**: Deep GNNs can cause node representations to become too similar.

**Solutions**:
- Use residual connections
- Limit depth (2-3 layers often sufficient)
- Use skip connections
- Apply normalization techniques

```python
class ResidualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2 + x1)  # Residual connection
        
        x3 = self.conv3(x2, edge_index)
        return x3
```

### Scalability

**Problem**: Large graphs don't fit in memory.

**Solutions**:
- Use GraphSAGE with neighborhood sampling
- Mini-batch training
- Subgraph sampling
- Use efficient libraries (DGL, PyTorch Geometric)

### Heterogeneous Graphs

**Problem**: Real-world graphs have multiple node/edge types.

**Solutions**:
- Use heterogeneous GNNs (RGCN, HAN)
- Separate embeddings for each type
- Type-specific message passing

## Practice Exercises

1. **Node Classification**: Classify nodes in Cora citation network using GCN
2. **Link Prediction**: Predict missing edges in a social network
3. **Graph Classification**: Classify molecular graphs by toxicity
4. **Recommendation System**: Build GNN-based recommender for movies
5. **Knowledge Graph**: Learn entity embeddings for question answering
6. **Fraud Detection**: Detect fraudulent transactions in transaction graph
7. **Traffic Prediction**: Predict traffic flow using road network graph

## Resources and Further Learning

### Books and Papers

1. **"Graph Neural Networks: A Review of Methods and Applications"** - Wu et al., 2020
2. **"Semi-Supervised Classification with Graph Convolutional Networks"** - Kipf & Welling, 2017 (GCN)
3. **"Graph Attention Networks"** - Veličković et al., 2018 (GAT)
4. **"Inductive Representation Learning on Large Graphs"** - Hamilton et al., 2017 (GraphSAGE)
5. **"How Powerful are Graph Neural Networks?"** - Xu et al., 2019 (GIN)

### Online Courses

1. **CS224W: Machine Learning with Graphs** - Stanford
   - [Course Website](http://web.stanford.edu/class/cs224w/)
   - Comprehensive coverage of GNNs, graph algorithms, and applications

2. **Graph Neural Networks** - DeepLearning.AI
   - Practical course on building GNNs

### Tutorials and Blogs

1. **PyTorch Geometric Tutorials**
   - [Official Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/index.html)
   - Comprehensive examples and use cases

2. **DGL Tutorials**
   - [DGL Tutorials](https://docs.dgl.ai/tutorials/index.html)
   - Step-by-step guides for various tasks

### Datasets

1. **Cora, CiteSeer, PubMed**: Citation networks
2. **OGB (Open Graph Benchmark)**: Large-scale graph datasets
3. **TUDataset**: Molecular and social network datasets
4. **Karate Club**: Small social network for testing

### Tools and Libraries

1. **PyTorch Geometric**: Most popular GNN library
2. **Deep Graph Library (DGL)**: Efficient and scalable
3. **Spektral**: Keras/TensorFlow GNN library
4. **StellarGraph**: Graph ML library with many algorithms

## Key Takeaways

1. **Graphs are Everywhere**: Many real-world problems have graph structure
2. **Message Passing**: Core mechanism of GNNs - aggregate information from neighbors
3. **Neighborhood Aggregation**: Combine information from neighbors (sum, mean, max, attention)
4. **Architecture Choice**: GCN for simple cases, GAT for attention, GraphSAGE for large graphs, GIN for maximum expressivity
5. **Libraries**: Use PyTorch Geometric or DGL for efficiency and ease of use
6. **Applications**: Social networks, molecules, knowledge graphs, recommendation systems, fraud detection
7. **Challenges**: Over-smoothing, scalability, heterogeneous graphs
8. **Best Practices**: Use residual connections, limit depth, normalize properly, sample neighborhoods for large graphs

---

**Next Steps**: Explore [Advanced Topics](graph-neural-networks-advanced-topics.md) for graph transformers, dynamic graphs, heterogeneous graphs, and graph generation.

