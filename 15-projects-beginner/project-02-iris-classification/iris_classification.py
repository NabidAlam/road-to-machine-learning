"""
Iris Flower Classification Project
A complete implementation for beginners
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_and_explore_data():
    """Load the Iris dataset and perform initial exploration"""
    print("=" * 60)
    print("STEP 1: Loading and Exploring Data")
    print("=" * 60)
    
    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create a DataFrame for easier exploration
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = pd.Categorical.from_codes(y, target_names)
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nSpecies Distribution:")
    print(df['species'].value_counts())
    
    return df, feature_names, target_names


def exploratory_data_analysis(df, feature_names):
    """Perform exploratory data analysis with visualizations"""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    
    # Pair plot to see relationships between features
    print("\nGenerating pair plot...")
    sns.pairplot(df, hue='species', diag_kind='hist')
    plt.suptitle('Pair Plot of Iris Features', y=1.02)
    plt.savefig('pair_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plots for each feature by species
    print("Generating box plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for idx, feature in enumerate(feature_names):
        row = idx // 2
        col = idx % 2
        sns.boxplot(data=df, x='species', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} by Species')
    plt.tight_layout()
    plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    correlation = df.iloc[:, :4].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_data(df):
    """Prepare data for modeling"""
    print("\n" + "=" * 60)
    print("STEP 3: Preparing Data for Modeling")
    print("=" * 60)
    
    # Separate features and target
    X = df.iloc[:, :4].values
    y = df['species'].cat.codes.values
    
    # Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test, target_names):
    """Train multiple classification models"""
    print("\n" + "=" * 60)
    print("STEP 4: Training Models")
    print("=" * 60)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}")
    
    return results


def compare_models(results):
    """Compare model performance"""
    print("\n" + "=" * 60)
    print("STEP 5: Comparing Models")
    print("=" * 60)
    
    # Visualize model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Comparison - Accuracy Scores', fontsize=14, fontweight='bold')
    plt.ylim([0.9, 1.0])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    return best_model_name


def plot_confusion_matrix(results, best_model_name, y_test, target_names):
    """Plot confusion matrix for the best model"""
    print("\n" + "=" * 60)
    print("STEP 6: Confusion Matrix")
    print("=" * 60)
    
    best_predictions = results[best_model_name]['predictions']
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def make_predictions(results, best_model_name, target_names):
    """Make predictions on new data"""
    print("\n" + "=" * 60)
    print("STEP 7: Making Predictions on New Data")
    print("=" * 60)
    
    # Example: Predict species for new measurements
    new_measurements = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 3.4, 5.4, 2.3],  # Virginica
        [5.9, 3.0, 4.2, 1.5]   # Versicolor
    ])
    
    best_model = results[best_model_name]['model']
    predictions = best_model.predict(new_measurements)
    
    print("\nPredictions for new measurements:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {target_names[pred]}")
    
    return predictions


def main():
    """Main function to run the complete project"""
    print("\n" + "=" * 60)
    print("IRIS FLOWER CLASSIFICATION PROJECT")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df, feature_names, target_names = load_and_explore_data()
    
    # Step 2: Exploratory data analysis
    exploratory_data_analysis(df, feature_names)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 4: Train models
    results = train_models(X_train, X_test, y_train, y_test, target_names)
    
    # Step 5: Compare models
    best_model_name = compare_models(results)
    
    # Step 6: Confusion matrix
    plot_confusion_matrix(results, best_model_name, y_test, target_names)
    
    # Step 7: Make predictions
    make_predictions(results, best_model_name, target_names)
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nCheck the generated images:")
    print("- pair_plot.png")
    print("- box_plots.png")
    print("- correlation_heatmap.png")
    print("- model_comparison.png")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()

