# Getting Started - Your First ML Project

Welcome! This guide will help you complete your first machine learning project in about 30 minutes.

## Quick Start: Iris Classification

The Iris Flower Classification project is the perfect first ML project. Follow these steps:

### Step 1: Set Up Environment

```bash
# Create virtual environment
python -m venv ml-env

# Activate (Windows)
ml-env\Scripts\activate

# Activate (Mac/Linux)
source ml-env/bin/activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 2: Run the Project

Navigate to the project directory:

```bash
cd 15-projects-beginner/project-02-iris-classification
```

Run the complete implementation:

```bash
python iris_classification.py
```

Or follow along with the step-by-step guide in `README.md`.

### Step 3: What You'll See

The script will:
1. Load and explore the Iris dataset
2. Create visualizations (pair plots, box plots, heatmaps)
3. Train 3 different models
4. Compare their performance
5. Show confusion matrix for the best model
6. Make predictions on new data

### Expected Output

- All models should achieve >95% accuracy
- You'll see 5 visualization images saved
- The best model will be identified
- Predictions for new flower measurements

## Understanding the Results

- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Shows which classes are confused with each other
- **Model Comparison**: Visual comparison of different algorithms

## Next Steps

1. Modify the code - try different models
2. Experiment with different train/test splits
3. Add your own features
4. Move to the next project: House Price Prediction

## Troubleshooting

**Import errors?**
- Make sure all packages are installed: `pip install -r requirements.txt`

**Plots not showing?**
- On some systems, you may need: `plt.show()` at the end
- Check if images are saved in the current directory

**Need help?**
- Check the project README for detailed explanations
- Review the code comments
- Open an issue on GitHub

## Why Start Here?

- **Simple Dataset**: Well-known, clean data
- **Clear Results**: Easy to understand outcomes
- **Complete Example**: Full working code provided
- **Quick Win**: See results in minutes

---

**Ready?** Go to `15-projects-beginner/project-02-iris-classification/` and start coding!

