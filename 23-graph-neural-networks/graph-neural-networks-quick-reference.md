# Graph Neural Networks Quick Reference

Quick reference for GNN formulas, architectures, and code.

## Key Formulas

### Message Passing
```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

### GCN
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

### GAT Attention
```
α_vu = softmax(LeakyReLU(a^T [W h_v || W h_u]))
```

## Architectures Comparison

| Architecture | Aggregation | Attention | Inductive |
|-------------|-------------|-----------|-----------|
| **GCN** | Mean | No | No |
| **GAT** | Weighted mean | Yes | No |
| **GraphSAGE** | Sample + Aggregate | No | Yes |
| **GIN** | Sum | No | Yes |

## Libraries

- **PyTorch Geometric**: `from torch_geometric.nn import GCNConv`
- **DGL**: `import dgl.nn as dglnn`
- **Spektral**: Keras/TensorFlow

