# Getting Started. Optional Iris Demo

This guide is an **optional ~30 minute demo** of a tiny sklearn classification run. It is not the default beginner path.

**Default for beginners:** start at [Module 00](00-prerequisites/README.md) and pass [Gate A](FOUNDATION_AND_JOB_READINESS.md#gate-a-after-module-00-stage-0). Iris does **not** replace Python, math, or those exit gates.

For the full curriculum map, stage order, and exit gates, read [START-HERE.md](START-HERE.md) and [FOUNDATION_AND_JOB_READINESS.md](FOUNDATION_AND_JOB_READINESS.md).

## Optional demo: Iris Classification

The Iris flower classification project is a small, clean toy run so you can see `fit` and metrics once. Follow these steps if you want the demo:

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

## Why this demo exists

- **Simple dataset**: Well-known, clean toy data
- **Clear results**: Easy to read metrics and plots
- **Complete example**: Full working code provided
- **Quick look**: See a sklearn loop in minutes. Then return to Module 00 / Gate A if you have not finished foundations.

---

**Optional demo:** `16-projects-beginner/project-02-iris-classification/`. **Default path:** [00-prerequisites/README.md](00-prerequisites/README.md) and Gate A.

