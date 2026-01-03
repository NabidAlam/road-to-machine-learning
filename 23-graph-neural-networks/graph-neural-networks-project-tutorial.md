# Graph Neural Networks Project Tutorial

Step-by-step tutorial: Node classification with GCN on Cora dataset.

## Project: Node Classification with GCN

### Objective

Classify research papers in the Cora citation network using Graph Convolutional Networks.

### Step 1: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
```

### Step 2: Load Dataset

```python
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"Nodes: {data.x.shape[0]}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Features: {data.x.shape[1]}")
print(f"Classes: {dataset.num_classes}")
```

### Step 3: Define GCN Model

```python
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### Step 4: Training

```python
model = GCN(input_dim=dataset.num_features, 
            hidden_dim=64, 
            output_dim=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    return acc.item()

# Train
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

