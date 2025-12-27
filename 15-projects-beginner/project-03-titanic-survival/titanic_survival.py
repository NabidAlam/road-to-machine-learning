"""
Titanic Survival Prediction Project
Predict which passengers survived the Titanic disaster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("Titanic Survival Prediction Project")
print("="*60)

# Note: You need to download the dataset from Kaggle
# https://www.kaggle.com/c/titanic/data
print("\nNote: Download train.csv from https://www.kaggle.com/c/titanic/data")
print("Place it in the data/ directory or update the path below.\n")

# Load data (update path as needed)
try:
    df = pd.read_csv('data/train.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset not found. Please download from Kaggle.")
    print("Creating sample structure for demonstration...")
    # Create sample data structure for demonstration
    df = pd.DataFrame({
        'PassengerId': range(1, 892),
        'Survived': np.random.randint(0, 2, 891),
        'Pclass': np.random.choice([1, 2, 3], 891),
        'Name': [f'Person {i}' for i in range(891)],
        'Sex': np.random.choice(['male', 'female'], 891),
        'Age': np.random.normal(30, 15, 891).clip(0, 80),
        'SibSp': np.random.randint(0, 4, 891),
        'Parch': np.random.randint(0, 3, 891),
        'Ticket': [f'Ticket{i}' for i in range(891)],
        'Fare': np.random.exponential(30, 891),
        'Cabin': [f'C{i}' if np.random.random() > 0.7 else np.nan for i in range(891)],
        'Embarked': np.random.choice(['S', 'C', 'Q'], 891)
    })
    print("Using sample data for demonstration purposes.")

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print("\n" + "="*60)
print("Missing Values Analysis")
print("="*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Basic statistics
print("\n" + "="*60)
print("Dataset Statistics")
print("="*60)
print(df.describe())

# Survival rate
print("\n" + "="*60)
print("Survival Analysis")
print("="*60)
survival_rate = df['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.2%}")
print(f"\nSurvival by Sex:")
print(df.groupby('Sex')['Survived'].mean())
print(f"\nSurvival by Pclass:")
print(df.groupby('Pclass')['Survived'].mean())

# Feature Engineering
print("\n" + "="*60)
print("Feature Engineering")
print("="*60)

# Create family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Extract title from name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Fill missing Age with median by Title
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing Fare with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Encode categorical variables
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])

le_embarked = LabelEncoder()
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

le_title = LabelEncoder()
df['Title_encoded'] = le_title.fit_transform(df['Title'])

le_age = LabelEncoder()
df['AgeGroup_encoded'] = le_age.fit_transform(df['AgeGroup'].astype(str))

# Select features for modeling
feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded']

X = df[feature_cols]
y = df['Survived']

print(f"\nSelected features: {feature_cols}")
print(f"Features shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

print("\n" + "="*60)
print("Model Training and Evaluation")
print("="*60)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Compare models
print("\n" + "="*60)
print("Model Comparison")
print("="*60)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})
print(comparison_df.to_string(index=False))

# Feature importance (Random Forest)
if 'Random Forest' in results:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*60)
    print("Feature Importance (Random Forest)")
    print("="*60)
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.show()

# Confusion matrix for best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model_name]['predictions']

cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Project Complete!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print("\nNext steps:")
print("1. Download actual Titanic dataset from Kaggle")
print("2. Submit predictions to Kaggle competition")
print("3. Try advanced feature engineering")
print("4. Experiment with ensemble methods")

