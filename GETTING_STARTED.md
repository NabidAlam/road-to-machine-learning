# Getting Started. Your First ML Project

This guide walks you through a first machine learning project in about 30 minutes.

For the full curriculum map, stage order, and exit gates, read [START-HERE.md](START-HERE.md) and [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md).

## Quick Start: Iris Classification

The Iris flower classification project is a small, clean first ML run. Follow these steps:

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
cd 16-projects-beginner/project-02-iris-classification
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

- On this clean toy dataset, models often land above about 95% accuracy. Treat that as a demo result, not a general ML promise.
- You will usually see visualization images saved in the project folder
- The script will pick a best model among the ones it trains
- It will show predictions for a few new flower measurements

## Understanding the Results

- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Shows which classes are confused with each other
- **Model Comparison**: Visual comparison of different algorithms

## Next Steps

1. Modify the code. Try different models
2. Experiment with different train/test splits
3. Add your own features
4. Move to the next project: House Price Prediction

## Troubleshooting

**Import errors?**
- Install from the repository root: `pip install -r requirements.txt` (run this from the top-level `road-to-machine-learning` folder, not inside a project subfolder)

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
- **Quick Win**. See results in minutes on a toy dataset.

---

**Ready?** Go to `16-projects-beginner/project-02-iris-classification/` and start coding!

