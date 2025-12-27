# Project 1: Handwritten Digit Recognition (MNIST)

Build a neural network to classify handwritten digits (0-9).

## Difficulty
Intermediate

## Time Estimate
3-5 days

## Skills You'll Practice
- Neural Networks
- Image Processing
- Deep Learning Basics
- Hyperparameter Tuning

## Learning Objectives

By completing this project, you will learn to:
- Build and train neural networks
- Preprocess image data
- Tune hyperparameters effectively
- Evaluate deep learning models
- Use data augmentation
- Achieve high accuracy (>99%)

## Dataset

**MNIST Dataset**
- Built into Keras/TensorFlow: `keras.datasets.mnist`
- 60,000 training images, 10,000 test images
- 28x28 grayscale images of handwritten digits (0-9)

## Project Steps

### Step 1: Load and Explore Data
- Load MNIST dataset
- Visualize sample images
- Check data distribution
- Normalize pixel values

### Step 2: Build Neural Network
- Create simple MLP (Multi-Layer Perceptron)
- Add hidden layers
- Choose activation functions
- Initialize model

### Step 3: Train Model
- Compile model (loss, optimizer, metrics)
- Train on training set
- Monitor training progress
- Use validation split

### Step 4: Evaluate Model
- Evaluate on test set
- Visualize predictions
- Analyze misclassified examples
- Calculate accuracy

### Step 5: Improve Model
- Add more layers
- Try different architectures
- Use dropout for regularization
- Tune hyperparameters
- Use data augmentation

### Step 6: Advanced Techniques
- Try CNN (Convolutional Neural Network)
- Compare MLP vs CNN
- Achieve >99% accuracy
- Visualize learned features

## Expected Deliverables

1. **Jupyter Notebook** with:
   - Data exploration
   - Model architecture
   - Training process
   - Evaluation results
   - Comparison of different models

2. **Trained Model**:
   - Saved model file
   - Can make predictions on new images

## Evaluation Metrics

- **Accuracy**: Primary metric (aim for >99%)
- **Confusion Matrix**: See which digits are confused
- **Per-class Accuracy**: Accuracy for each digit

## Tips

- Start with simple MLP, then try CNN
- Normalize pixel values to [0, 1]
- Use early stopping to prevent overfitting
- Try different optimizers (Adam, SGD)
- Data augmentation can help improve accuracy
- Visualize misclassified examples to understand errors

## Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras MNIST Tutorial](https://keras.io/examples/vision/mnist_convnet/)
- [TensorFlow MNIST Guide](https://www.tensorflow.org/tutorials/quickstart/beginner)

## Extensions

- Try different architectures (ResNet, etc.)
- Build web app to upload and classify images
- Try on other digit datasets (USPS, SVHN)
- Implement from scratch (without frameworks)

## Next Steps

After completing this project:
- Move to CIFAR-10 for color images
- Try more complex architectures
- Experiment with transfer learning
- Move to [Project 2: Customer Churn](../project-02-customer-churn/README.md)

