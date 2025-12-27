# Phase 9: Neural Networks Basics

Introduction to neural networks - the foundation of deep learning.

##  What You'll Learn

- Perceptron and Multi-Layer Perceptron (MLP)
- Activation Functions
- Backpropagation
- Gradient Descent Variants
- Building Neural Networks from Scratch

##  Topics Covered

### 1. Perceptron
- **Single Perceptron**: Basic building block
- **Linear Classification**: Can only solve linearly separable problems
- **Limitations**: XOR problem

### 2. Multi-Layer Perceptron (MLP)
- **Hidden Layers**: Enable non-linear learning
- **Architecture**: Input → Hidden → Output
- **Universal Approximation**: Can approximate any function

### 3. Activation Functions
- **Sigmoid**: S-shaped curve (0 to 1)
- **Tanh**: Hyperbolic tangent (-1 to 1)
- **ReLU**: Rectified Linear Unit (most common)
- **Softmax**: For multi-class classification
- **When to use each**: Different use cases

### 4. Backpropagation
- **Forward Pass**: Compute predictions
- **Backward Pass**: Compute gradients
- **Chain Rule**: How gradients flow backward
- **Understanding**: Key to training neural networks

### 5. Gradient Descent
- **Batch Gradient Descent**: Use all data
- **Stochastic Gradient Descent (SGD)**: One sample at a time
- **Mini-batch Gradient Descent**: Small batches (most common)
- **Optimizers**: Adam, RMSprop, AdaGrad

### 6. Loss Functions
- **Mean Squared Error (MSE)**: For regression
- **Cross-Entropy**: For classification
- **Binary Cross-Entropy**: For binary classification

##  Learning Objectives

By the end of this module, you should be able to:
- Understand how neural networks work
- Build a neural network from scratch
- Implement backpropagation
- Choose appropriate activation functions
- Train neural networks effectively

##  Projects

1. **Build Perceptron from Scratch**: Implement basic perceptron
2. **Build MLP from Scratch**: Implement multi-layer network
3. **MNIST Classification**: Classify handwritten digits with MLP
4. **Compare Optimizers**: Test different optimization algorithms

##  Key Concepts

- **Weights and Biases**: Parameters learned during training
- **Forward Propagation**: Computing predictions
- **Backward Propagation**: Computing gradients
- **Learning Rate**: How fast to learn (critical hyperparameter)
- **Epochs**: Full pass through training data
- **Batch Size**: Number of samples per update

##  Additional Resources

- [Neural Networks and Deep Learning (Free Book)](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Keras Documentation](https://keras.io/)

---

**Previous Phase:** [08-unsupervised-learning](../08-unsupervised-learning/README.md)  
**Next Phase:** [10-deep-learning-frameworks](../10-deep-learning-frameworks/README.md)

