# Data Science & Machine Learning Cheatsheet

Quick reference for common syntax and operations used in everyday data science and machine learning work. Covers Python, NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow, OpenCV, FastAPI, and more.

## Table of Contents

- [NumPy](#numpy)
- [Pandas](#pandas)
- [Matplotlib & Seaborn](#matplotlib--seaborn)
- [Scikit-learn](#scikit-learn)
- [PyTorch](#pytorch)
- [TensorFlow/Keras](#tensorflowkeras)
- [OpenCV](#opencv)
- [Hyperparameter Tuning & Model Optimization](#hyperparameter-tuning--model-optimization)
- [FastAPI & Web Development](#fastapi--web-development)
- [File Operations](#file-operations)
- [List & Dictionary Operations](#list--dictionary-operations)
- [String Operations](#string-operations)
- [Date & Time](#date--time)
- [Useful Functions](#useful-functions)

---

## NumPy

### Creating Arrays

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Output: [1 2 3 4 5]

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Output: [[1 2 3]
#          [4 5 6]]

zeros = np.zeros((3, 4))
print(zeros)
# Output: [[0. 0. 0. 0.]
#          [0. 0. 0. 0.]
#          [0. 0. 0. 0.]]

ones = np.ones((2, 3))
print(ones)
# Output: [[1. 1. 1.]
#          [1. 1. 1.]]

arange = np.arange(0, 10, 2)
print(arange)
# Output: [0 2 4 6 8]

linspace = np.linspace(0, 1, 5)
print(linspace)
# Output: [0.   0.25 0.5  0.75 1.  ]

random = np.random.rand(3, 3)
print(random)
# Output: [[0.37454012 0.95071431 0.73199394]
#          [0.59865848 0.15601864 0.15599452]
#          [0.05808361 0.86617615 0.60111501]]

randint = np.random.randint(0, 10, (3, 3))
print(randint)
# Output: [[3 7 9]
#          [3 5 2]
#          [4 7 6]]
```

### Array Operations

```python
# Shape and dimensions
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)   # Output: (2, 3)
print(arr.ndim)    # Output: 2
print(arr.size)    # Output: 6
print(arr.dtype)   # Output: int64

# Reshaping
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, 3)
print(reshaped)
# Output: [[1 2 3]
#          [4 5 6]]

flattened = arr.flatten()
print(flattened)  # Output: [1 2 3 4 5 6]

# Indexing and slicing
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])      # Output: 1 - First element
print(arr[-1])     # Output: 5 - Last element
print(arr[1:4])    # Output: [2 3 4] - Slice
print(arr[arr > 3])  # Output: [4 5] - Boolean indexing

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d[0, 1])   # Output: 2 - 2D indexing
print(arr2d[:, 1])   # Output: [2 5] - All rows, column 1
print(arr2d[1, :])   # Output: [4 5 6] - Row 1, all columns

# Mathematical operations
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))        # Output: 15
print(np.mean(arr))       # Output: 3.0
print(np.std(arr))        # Output: 1.4142135623730951
print(np.var(arr))        # Output: 2.0
print(np.min(arr))        # Output: 1
print(np.max(arr))        # Output: 5
print(np.median(arr))     # Output: 3.0
print(np.percentile(arr, 50))  # Output: 3.0

# Element-wise operations
arr = np.array([1, 2, 3, 4])
print(arr + 1)      # Output: [2 3 4 5]
print(arr * 2)      # Output: [2 4 6 8]
print(arr ** 2)     # Output: [ 1  4  9 16]
print(np.sqrt(arr)) # Output: [1.         1.41421356 1.73205081 2.        ]
print(np.exp([1, 2]))  # Output: [2.71828183 7.3890561 ]
print(np.log([1, np.e]))  # Output: [0. 1.]

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)  # Output: [5 7 9] - Element-wise addition
print(arr1 * arr2)  # Output: [ 4 10 18] - Element-wise multiplication

# Matrix multiplication
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(np.dot(arr1, arr2))
# Output: [[19 22]
#          [43 50]]

print(arr1 @ arr2)  # Same as dot product
# Output: [[19 22]
#          [43 50]]
```

### Useful NumPy Functions

```python
np.concatenate([arr1, arr2])
np.vstack([arr1, arr2])  # Vertical stack
np.hstack([arr1, arr2])  # Horizontal stack
np.where(condition, x, y)  # Conditional
np.unique(arr)  # Unique values
np.sort(arr)  # Sort
np.argsort(arr)  # Sort indices
```

---

## Pandas

### Creating DataFrames

```python
import pandas as pd

# From dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
# Output:    A  B
#         0  1  4
#         1  2  5
#         2  3  6

# From CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', index_col=0)
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# From Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# From JSON
df = pd.read_json('file.json')

# From dictionary of lists
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
```

### Viewing Data

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.head())        # First 5 rows
# Output:    A  B
#         0  1  4
#         1  2  5
#         2  3  6

print(df.shape)         # Output: (3, 2) - (rows, columns)
print(df.columns)       # Output: Index(['A', 'B'], dtype='object')
print(df.dtypes)        # Output: A    int64
                          #         B    int64
                          #         dtype: object

print(df.describe())
# Output:              A         B
#         count  3.000000  3.000000
#         mean   2.000000  5.000000
#         std    1.000000  1.000000
#         min    1.000000  4.000000
#         25%    1.500000  4.500000
#         50%    2.000000  5.000000
#         75%    2.500000  5.500000
#         max    3.000000  6.000000
```

### Selecting Data

```python
# Single column (returns Series)
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df['A'])
# Output: 0    1
#         1    2
#         2    3
#         Name: A, dtype: int64

# Multiple columns (returns DataFrame)
print(df[['A', 'B']])
# Output:    A  B
#         0  1  4
#         1  2  5
#         2  3  6

# Rows by index
df.iloc[0]           # First row
df.iloc[0:5]        # First 5 rows
df.iloc[0, 1]       # Row 0, column 1
df.iloc[0:5, 1:3]   # Rows 0-4, columns 1-2

# Rows by label
df.loc['index_name']
df.loc['row1':'row5', 'col1':'col3']

# Boolean indexing
df = pd.DataFrame({'A': [1, 5, 8, 3], 'B': [10, 20, 30, 40]})
print(df[df['A'] > 5])
# Output:    A   B
#         2  8  30

print(df[(df['A'] > 2) & (df['B'] < 35)])
# Output:    A   B
#         1  5  20
#         2  8  30

print(df[df['A'].isin([1, 8])])
# Output:    A   B
#         0  1  10
#         2  8  30
```

### Data Manipulation

```python
# Adding/Removing columns
df['new_col'] = values
df.drop('column', axis=1)  # Drop column
df.drop([0, 1], axis=0)     # Drop rows

# Renaming
df.rename(columns={'old': 'new'})
df.rename(index={0: 'first'})

# Sorting
df.sort_values('column')
df.sort_values('column', ascending=False)
df.sort_values(['col1', 'col2'])

# Grouping
df = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'A'],
    'Value1': [10, 20, 30, 40, 50],
    'Value2': [1, 2, 3, 4, 5]
})
print(df.groupby('Category').mean())
# Output:          Value1  Value2
#         Category
#         A         26.666667  2.666667
#         B         35.000000  3.500000

print(df.groupby('Category').sum())
# Output:          Value1  Value2
#         Category
#         A           80      8
#         B           70      7

print(df.groupby('Category').agg({'Value1': 'mean', 'Value2': 'sum'}))
# Output:          Value1  Value2
#         Category
#         A         26.666667      8
#         B         35.000000      7

# Merging
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')
pd.concat([df1, df2])  # Concatenate
df1.join(df2)  # Join on index
```

### Handling Missing Data

```python
df.isna()        # Check for NaN
df.isnull()      # Same as isna()
df.notna()       # Check for non-NaN
df.dropna()      # Drop rows with NaN
df.dropna(axis=1)  # Drop columns with NaN
df.fillna(0)     # Fill NaN with 0
df.fillna(df.mean())  # Fill with mean
df['col'].fillna(df['col'].median())
```

### Data Types

```python
df['col'].astype(int)
df['col'].astype(float)
df['col'].astype(str)
df['col'].astype('category')
pd.to_datetime(df['date_col'])
```

### String Operations

```python
df['col'].str.lower()
df['col'].str.upper()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.contains('text')
df['col'].str.split(' ')
df['col'].str.len()
```

### Aggregations

```python
df.sum()
df.mean()
df.median()
df.std()
df.min()
df.max()
df.count()
df.nunique()  # Number of unique values
```

### Saving Data

```python
df.to_csv('output.csv')
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx')
df.to_json('output.json')
df.to_pickle('output.pkl')
```

---

## Matplotlib & Seaborn

### Basic Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot(x, y)
plt.plot(x, y, label='Line', color='blue', linestyle='--', linewidth=2)

# Scatter plot
plt.scatter(x, y, s=50, alpha=0.5, c='red')

# Bar plot
plt.bar(x, y)
plt.barh(x, y)  # Horizontal

# Histogram
plt.hist(data, bins=30)

# Box plot
plt.boxplot(data)

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)

# Styling
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Seaborn

```python
# Distribution plots
sns.distplot(data)
sns.histplot(data, kde=True)

# Scatter plot
sns.scatterplot(x='col1', y='col2', data=df, hue='col3')

# Line plot
sns.lineplot(x='col1', y='col2', data=df)

# Bar plot
sns.barplot(x='col1', y='col2', data=df)

# Box plot
sns.boxplot(x='col1', y='col2', data=df)

# Violin plot
sns.violinplot(x='col1', y='col2', data=df)

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Pair plot
sns.pairplot(df, hue='target')

# Count plot
sns.countplot(x='column', data=df)

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')
```

---

## Scikit-learn

### Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Min-Max scaling
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### Models

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification
model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100)
model = DecisionTreeClassifier()
model = SVC()
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report

y_test = [0, 1, 0, 1, 0]
predictions = [0, 1, 0, 1, 1]

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
# Output: Accuracy: 0.8

print(f"Precision: {precision_score(y_test, predictions)}")
# Output: Precision: 1.0

print(f"Recall: {recall_score(y_test, predictions)}")
# Output: Recall: 0.5

print(f"F1 Score: {f1_score(y_test, predictions)}")
# Output: F1 Score: 0.6666666666666666

print(confusion_matrix(y_test, predictions))
# Output: [[2 0]
#          [1 2]]

# Regression
from sklearn.metrics import mean_squared_error, r2_score

y_test = [3, 5, 7, 9]
predictions = [3.1, 4.9, 7.2, 8.8]

mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")  # Output: MSE: 0.025

rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")  # Output: RMSE: 0.15811388300841898

r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2}")  # Output: R² Score: 0.9983333333333333
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

---

## PyTorch

### Creating Tensors

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create tensors
tensor = torch.tensor([1, 2, 3])
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.rand(3, 3)  # Random 0-1
randn = torch.randn(3, 3)  # Random normal
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)

# From NumPy
tensor = torch.from_numpy(np_array)
np_array = tensor.numpy()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
```

### Tensor Operations

```python
# Shape and properties
tensor.shape
tensor.size()
tensor.dim()
tensor.dtype
tensor.device

# Reshaping
tensor.view(2, 3)
tensor.reshape(2, 3)
tensor.flatten()
tensor.squeeze()  # Remove dims of size 1
tensor.unsqueeze(0)  # Add dimension

# Indexing and slicing
tensor[0]
tensor[0:5]
tensor[:, 1]
tensor[..., 0]  # Last dimension

# Mathematical operations
torch.sum(tensor)
torch.mean(tensor)
torch.std(tensor)
torch.max(tensor)
torch.min(tensor)
torch.abs(tensor)
torch.sqrt(tensor)
torch.exp(tensor)
torch.log(tensor)

# Element-wise operations
tensor1 + tensor2
tensor1 * tensor2
torch.mul(tensor1, tensor2)
torch.div(tensor1, tensor2)
torch.pow(tensor, 2)

# Matrix operations
torch.matmul(tensor1, tensor2)
tensor1 @ tensor2
torch.transpose(tensor, 0, 1)
tensor.T
```

### Neural Network Basics

```python
# Define a simple network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net().to(device)
```

### Common Layers

```python
# Linear/FC layers
nn.Linear(in_features, out_features)

# Convolutional layers
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.Conv1d(in_channels, out_channels, kernel_size)

# Pooling
nn.MaxPool2d(kernel_size, stride=None)
nn.AvgPool2d(kernel_size, stride=None)
nn.AdaptiveAvgPool2d(output_size)

# Normalization
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Activation functions
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.LeakyReLU(negative_slope=0.01)
nn.Softmax(dim=1)

# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)

# Recurrent layers
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size, num_layers)
```

### Training Loop

```python
# Loss function
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
criterion = nn.BCELoss()
criterion = nn.NLLLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Training step
model.train()
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    output = model(input)
    predictions = torch.argmax(output, dim=1)
```

### Data Loading

```python
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate
for batch_data, batch_labels in dataloader:
    batch_data = batch_data.to(device)
    batch_labels = batch_labels.to(device)
    # Training code
```

### Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')
torch.save(model, 'model_full.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))
model = torch.load('model_full.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## TensorFlow/Keras

### Creating Tensors

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create tensors
tensor = tf.constant([1, 2, 3])
zeros = tf.zeros((3, 4))
ones = tf.ones((2, 3))
rand = tf.random.normal((3, 3))
rand_uniform = tf.random.uniform((3, 3), 0, 1)
arange = tf.range(0, 10, 2)

# From NumPy
tensor = tf.constant(np_array)
np_array = tensor.numpy()

# Variable
var = tf.Variable([1.0, 2.0])
```

### Tensor Operations

```python
# Shape and properties
tensor.shape
tf.rank(tensor)
tensor.dtype
tensor.device

# Reshaping
tf.reshape(tensor, (2, 3))
tensor.reshape((2, 3))
tf.expand_dims(tensor, axis=0)
tf.squeeze(tensor)

# Mathematical operations
tf.reduce_sum(tensor)
tf.reduce_mean(tensor)
tf.reduce_max(tensor)
tf.reduce_min(tensor)
tf.abs(tensor)
tf.sqrt(tensor)
tf.exp(tensor)
tf.log(tensor)

# Element-wise
tf.add(tensor1, tensor2)
tf.multiply(tensor1, tensor2)
tf.divide(tensor1, tensor2)
tensor1 + tensor2
tensor1 * tensor2

# Matrix operations
tf.matmul(tensor1, tensor2)
tf.transpose(tensor)
tf.linalg.inv(tensor)
```

### Building Models

```python
# Sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

### Common Layers

```python
# Dense/FC
layers.Dense(units, activation=None)

# Convolutional
layers.Conv2D(filters, kernel_size, strides=1, padding='valid')
layers.Conv1D(filters, kernel_size)

# Pooling
layers.MaxPooling2D(pool_size=2, strides=2)
layers.AveragePooling2D(pool_size=2)
layers.GlobalAveragePooling2D()

# Normalization
layers.BatchNormalization()
layers.LayerNormalization()

# Activation
layers.Activation('relu')
layers.ReLU()
layers.LeakyReLU(alpha=0.01)

# Dropout
layers.Dropout(rate=0.5)

# Recurrent
layers.LSTM(units, return_sequences=False)
layers.GRU(units)
layers.SimpleRNN(units)

# Embedding
layers.Embedding(input_dim, output_dim)
```

### Compiling and Training

```python
# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Or with objects
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Training
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1
)

# Evaluation
loss, accuracy = model.evaluate(x_test, y_test)

# Prediction
predictions = model.predict(x_test)
predictions = model(x_test)  # Eager execution
```

### Callbacks

```python
# Early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# Reduce learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)

# Use in training
model.fit(..., callbacks=[early_stop, checkpoint, reduce_lr])
```

### Data Preprocessing

```python
# Image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    'train_dir',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Text preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=100)
```

### Saving and Loading

```python
# Save model
model.save('model.h5')
model.save('model_directory')  # SavedModel format

# Load model
model = keras.models.load_model('model.h5')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')

# Save architecture
json_string = model.to_json()
model = keras.models.model_from_json(json_string)
```

---

## OpenCV

### Image Reading and Writing

```python
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg')
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# Write image
cv2.imwrite('output.jpg', img)

# Display image
cv2.imshow('Window Name', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image properties
height, width = img.shape[:2]
img.shape
img.dtype
img.size
```

### Color Spaces

```python
# Convert color spaces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split channels
b, g, r = cv2.split(img)
h, s, v = cv2.split(hsv)

# Merge channels
img = cv2.merge([b, g, r])
```

### Image Operations

```python
# Resize
resized = cv2.resize(img, (width, height))
resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# Crop
cropped = img[y1:y2, x1:x2]

# Rotate
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# Flip
flipped_h = cv2.flip(img, 1)  # Horizontal
flipped_v = cv2.flip(img, 0)  # Vertical
flipped_both = cv2.flip(img, -1)  # Both

# Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(img, M, (w, h))
```

### Drawing

```python
# Draw line
cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

# Draw rectangle
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Filled

# Draw circle
cv2.circle(img, (x, y), radius, (0, 255, 0), thickness=2)
cv2.circle(img, (x, y), radius, (0, 255, 0), -1)  # Filled

# Draw ellipse
cv2.ellipse(img, (x, y), (w, h), angle, 0, 360, (0, 255, 0), 2)

# Draw text
cv2.putText(img, 'Text', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

# Draw polygon
pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
cv2.polylines(img, [pts], True, (0, 255, 0), 2)
```

### Image Filtering

```python
# Blur
blur = cv2.blur(img, (5, 5))
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Edge detection
edges = cv2.Canny(img, 100, 200)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Morphological operations
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

### Thresholding

```python
# Binary threshold
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Otsu's threshold
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Contours

```python
# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Contour properties
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)
x, y, w, h = cv2.boundingRect(contour)
(x, y), radius = cv2.minEnclosingCircle(contour)
```

### Feature Detection

```python
# Corner detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, 
                                   minDistance=10)

# SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, None)

# ORB
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

# FAST
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
```

### Video Processing

```python
# Read video
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
out.write(frame)
out.release()
```

### Image Arithmetic

```python
# Add images
result = cv2.add(img1, img2)
result = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

# Bitwise operations
bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_xor = cv2.bitwise_xor(img1, img2)
bitwise_not = cv2.bitwise_not(img1)

# Masking
masked = cv2.bitwise_and(img, img, mask=mask)
```

### Histogram

```python
# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Histogram equalization
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
```

---

## Hyperparameter Tuning & Model Optimization

### Scikit-learn Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Randomized Search (faster for large parameter spaces)
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### Optuna - Advanced Hyperparameter Optimization

```python
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Define objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Cross-validation score
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Create study and optimize
study = optuna.create_study(direction='maximize', study_name='rf_optimization')
study.optimize(objective, n_trials=100, n_jobs=-1)

# Get best parameters
best_params = study.best_params
best_score = study.best_value
best_trial = study.best_trial

# Visualize optimization
plot_optimization_history(study).show()
plot_param_importances(study).show()

# Pruning (early stopping)
def objective_with_pruning(trial):
    # ... model setup ...
    for step in range(n_steps):
        # Training step
        intermediate_value = train_step()
        trial.report(intermediate_value, step)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_value

study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
```

### Hyperopt - Bayesian Optimization

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [3, 5, 7, None]),
    'min_samples_split': hp.uniform('min_samples_split', 2, 10),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3))
}

# Objective function
def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return {'loss': -score, 'status': STATUS_OK}

# Optimize
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
```

### Ray Tune - Distributed Hyperparameter Tuning

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=42
    )
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    tune.report(accuracy=score)

# Define search space
config = {
    'n_estimators': tune.choice([50, 100, 200]),
    'max_depth': tune.choice([3, 5, 7, None])
}

# Run optimization
analysis = tune.run(
    train_model,
    config=config,
    num_samples=100,
    scheduler=ASHAScheduler(metric='accuracy', mode='max'),
    resources_per_trial={'cpu': 2}
)

best_config = analysis.get_best_config(metric='accuracy', mode='max')
```

### Model Optimization Techniques

#### Learning Rate Scheduling

```python
# PyTorch Learning Rate Scheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

# Step LR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential LR
scheduler = ExponentialLR(optimizer, gamma=0.95)

# Reduce on Plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# In training loop
for epoch in range(epochs):
    train()
    val_loss = validate()
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # or scheduler.step() for others

# TensorFlow/Keras Learning Rate Scheduling
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# Reduce on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Custom schedule
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

lr_scheduler = LearningRateScheduler(scheduler)

model.fit(..., callbacks=[reduce_lr, lr_scheduler])
```

#### Early Stopping

```python
# Scikit-learn (manual)
from sklearn.model_selection import validation_curve

train_scores, val_scores = validation_curve(
    model, X_train, y_train, param_name='max_depth',
    param_range=[1, 3, 5, 7, 9], cv=5
)

# PyTorch Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True
        return False

# TensorFlow/Keras Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=0.001
)
```

#### Regularization Techniques

```python
# L1/L2 Regularization
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# L1 (Lasso)
lasso = Lasso(alpha=0.1)

# L2 (Ridge)
ridge = Ridge(alpha=0.1)

# Elastic Net (L1 + L2)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Dropout (PyTorch)
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(256, 10)
)

# Dropout (TensorFlow/Keras)
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Batch Normalization
# PyTorch
nn.BatchNorm1d(256)
nn.BatchNorm2d(64)

# TensorFlow/Keras
layers.BatchNormalization()
```

#### Model Ensembling

```python
# Voting Classifier
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('lr', LogisticRegression())
    ],
    voting='soft'  # or 'hard'
)

# Stacking
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC()),
        ('lr', LogisticRegression())
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Blending
# Train base models on training set
# Make predictions on validation set
# Train meta-model on validation predictions
```

### Advanced Optimization Techniques

#### Gradient Clipping

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# In training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# TensorFlow/Keras
optimizer = keras.optimizers.Adam(clipnorm=1.0)
optimizer = keras.optimizers.Adam(clipvalue=0.5)
```

#### Mixed Precision Training

```python
# PyTorch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# TensorFlow/Keras
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
```

#### Model Pruning

```python
# PyTorch Pruning
import torch.nn.utils.prune as prune

# L1 Unstructured Pruning
prune.l1_unstructured(module, name='weight', amount=0.2)

# Global Pruning
parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# TensorFlow Model Optimization
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

#### Quantization

```python
# PyTorch Quantization
import torch.quantization

# Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# TensorFlow Quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

### Hyperparameter Tuning Best Practices

```python
# 1. Define reasonable search spaces
# Too narrow: miss good solutions
# Too wide: waste computation

# 2. Use appropriate search strategy
# Grid Search: Small, discrete spaces
# Random Search: Large spaces, faster
# Bayesian Optimization: Expensive evaluations

# 3. Cross-validation for robust evaluation
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

# 4. Track experiments
import mlflow

mlflow.set_experiment("hyperparameter_tuning")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)
    mlflow.sklearn.log_model(best_model, "model")

# 5. Use validation set properly
# Train set: Training models
# Validation set: Tuning hyperparameters
# Test set: Final evaluation (only once!)

# 6. Consider computational budget
# Start with random search (fast)
# Refine with grid/bayesian (thorough)

# 7. Monitor overfitting
# Track train vs validation performance
# Use early stopping
# Regularize appropriately
```

### Modern Optimization Tools

```python
# Weights & Biases (W&B) for tracking
import wandb

wandb.init(project="ml-optimization")

config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10
}

wandb.config.update(config)

for epoch in range(epochs):
    train_loss = train()
    val_loss = validate()
    
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch
    })

# MLflow for experiment tracking
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_optimization")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

### Quick Reference: Common Hyperparameters

```python
# Random Forest
{
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# XGBoost
{
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Neural Networks
{
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'dropout': [0.1, 0.3, 0.5],
    'hidden_units': [64, 128, 256, 512],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}

# CNN
{
    'filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'pool_size': [2, 3],
    'dense_units': [64, 128, 256]
}
```

---

## FastAPI & Web Development

### FastAPI Basics

```python
from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

# Create app
app = FastAPI(
    title="ML API",
    description="Machine Learning API for predictions",
    version="1.0.0"
)

# Basic route
@app.get("/")
def read_root():
    return {"message": "Welcome to ML API"}

# Route with path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# Route with query parameters
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# POST request with Pydantic model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
def create_item(item: Item):
    return item

# Run server
# uvicorn main:app --reload
```

### Pydantic Models for Data Validation

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# Basic model
class User(BaseModel):
    name: str
    email: str
    age: int

# Model with validation
class PredictionRequest(BaseModel):
    feature1: float = Field(..., gt=0, description="Feature 1 must be positive")
    feature2: float = Field(..., ge=0, le=100, description="Feature 2 between 0-100")
    feature3: Optional[str] = None
    
    @validator('feature1')
    def validate_feature1(cls, v):
        if v > 1000:
            raise ValueError('Feature1 too large')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "feature1": 5.5,
                "feature2": 50.0,
                "feature3": "category_a"
            }
        }

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
```

### ML Model Deployment with FastAPI

```python
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI(title="House Price Prediction API")

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int

@app.post("/predict", response_model=dict)
async def predict_price(features: HouseFeatures):
    try:
        # Convert to array
        feature_array = np.array([[
            features.bedrooms,
            features.bathrooms,
            features.sqft_living,
            features.sqft_lot,
            features.floors,
            features.waterfront,
            features.view,
            features.condition,
            features.grade,
            features.sqft_above,
            features.sqft_basement,
            features.yr_built,
            features.yr_renovated
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        
        return {
            "predicted_price": float(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

### File Upload for ML Models

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import io

app = FastAPI()

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    # Read CSV file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Make predictions
    predictions = model.predict(df.values)
    
    # Add predictions to dataframe
    df['predictions'] = predictions
    
    # Convert to CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    
    return {"predictions": predictions.tolist()}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    
    # Process image (e.g., with OpenCV or PIL)
    # ... image processing code ...
    
    # Make prediction
    prediction = image_model.predict(processed_image)
    
    return {"prediction": prediction}
```

### Error Handling

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc), "type": "ValueError"}
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

# HTTP exception handler
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

# Usage
@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id < 0:
        raise HTTPException(status_code=400, detail="Item ID must be positive")
    if item_id > 1000:
        raise ValueError("Item ID too large")
    return {"item_id": item_id}
```

### Authentication & Security

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.get("/protected")
def protected_route(payload: dict = Depends(verify_token)):
    return {"message": "This is a protected route", "user": payload.get("sub")}
```

### CORS and Middleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)

# Custom middleware
from starlette.middleware.base import BaseHTTPMiddleware
import time

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(TimingMiddleware)
```

### Background Tasks

```python
from fastapi import FastAPI, BackgroundTasks
import asyncio

app = FastAPI()

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

def train_model_async():
    # Long-running ML training task
    # ... training code ...
    pass

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model_async)
    return {"message": "Training started in background"}

@app.post("/send-notification")
def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification will be sent"}
```

### Database Integration

```python
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "sqlite:///./ml_predictions.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(String)
    prediction = Column(String)
    timestamp = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict-with-db")
def predict_with_db(features: HouseFeatures, db: Session = Depends(get_db)):
    # Make prediction
    prediction = model.predict([...])
    
    # Save to database
    db_prediction = Prediction(
        input_data=str(features.dict()),
        prediction=str(prediction),
        timestamp=datetime.now().isoformat()
    )
    db.add(db_prediction)
    db.commit()
    
    return {"prediction": prediction, "id": db_prediction.id}
```

### Testing FastAPI

```python
from fastapi.testclient import TestClient
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Test client
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_predict():
    response = client.post(
        "/predict",
        json={
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_living": 2000
        }
    )
    assert response.status_code == 200
    assert "predicted_price" in response.json()
```

### Deployment

```python
# Dockerfile example
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# docker-compose.yml
"""
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pkl
"""

# Run with uvicorn
# uvicorn main:app --host 0.0.0.0 --port 8000
# uvicorn main:app --reload  # Development mode
# uvicorn main:app --workers 4  # Production with multiple workers
```

### API Documentation

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="ML Prediction API",
    description="""
    ## Machine Learning Prediction API
    
    This API provides endpoints for:
    * House price prediction
    * Image classification
    * Batch predictions
    
    ## Authentication
    Use Bearer token in Authorization header
    """,
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ML API",
        version="1.0.0",
        description="Custom API documentation",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### Best Practices for Data Scientists

```python
# 1. Always validate input data
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=10, max_items=10)
    
    @validator('features')
    def validate_features(cls, v):
        if any(x < 0 for x in v):
            raise ValueError('All features must be non-negative')
        return v

# 2. Handle model loading errors
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model not found")

# 3. Add logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(data: PredictionInput):
    logger.info(f"Received prediction request: {data}")
    try:
        result = model.predict([data.features])
        logger.info(f"Prediction successful: {result}")
        return {"prediction": result[0]}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. Version your API
@app.get("/v1/predict")
def predict_v1():
    pass

@app.get("/v2/predict")
def predict_v2():
    pass

# 5. Add rate limiting (use slowapi)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, data: PredictionInput):
    return {"prediction": model.predict([data.features])[0]}
```

### Learning Resources

#### Official Documentation
- **FastAPI Official Docs**: https://fastapi.tiangolo.com/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Uvicorn Documentation**: https://www.uvicorn.org/

#### Tutorials & Courses
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Real Python FastAPI Guide**: https://realpython.com/fastapi-python-web-apis/
- **Full Stack FastAPI Template**: https://github.com/tiangolo/full-stack-fastapi-template


#### Books
- **"Building Data Science Applications with FastAPI"** by Francois Voron
- **"FastAPI Modern Python Web Development"** by Bill Lubanovic

#### Best Practices Guides
- **FastAPI Best Practices**: https://github.com/zhanymkanov/fastapi-best-practices
- **API Design Guidelines**: https://restfulapi.net/
- **12 Factor App**: https://12factor.net/

#### Deployment Platforms
- **Heroku**: https://devcenter.heroku.com/articles/getting-started-with-python
- **AWS Lambda**: https://aws.amazon.com/lambda/
- **Google Cloud Run**: https://cloud.google.com/run
- **DigitalOcean App Platform**: https://www.digitalocean.com/products/app-platform
- **Railway**: https://railway.app/
- **Render**: https://render.com/

#### Additional Tools
- **Postman**: API testing - https://www.postman.com/
- **Swagger UI**: Auto-generated API docs (built into FastAPI)
- **Docker**: Containerization - https://www.docker.com/
- **Nginx**: Reverse proxy - https://www.nginx.com/

### Quick Reference: Common Patterns

```python
# Pattern 1: Simple ML API
@app.post("/predict")
def predict(features: Features):
    prediction = model.predict([features.to_array()])
    return {"prediction": prediction[0]}

# Pattern 2: Batch Prediction
@app.post("/predict-batch")
def predict_batch(features_list: List[Features]):
    predictions = model.predict([f.to_array() for f in features_list])
    return {"predictions": predictions.tolist()}

# Pattern 3: Async Prediction
@app.post("/predict-async")
async def predict_async(features: Features):
    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(
        None, model.predict, [[features.to_array()]]
    )
    return {"prediction": prediction[0]}

# Pattern 4: Model Versioning
MODEL_VERSION = "v1.2.0"

@app.get("/model-info")
def model_info():
    return {
        "version": MODEL_VERSION,
        "features": model.n_features_in_,
        "model_type": type(model).__name__
    }
```

---

## File Operations

### Reading Files

```python
# Text file
with open('file.txt', 'r') as f:
    content = f.read()
    lines = f.readlines()

# CSV
import csv
with open('file.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# JSON
import json
with open('file.json', 'r') as f:
    data = json.load(f)
```

### Writing Files

```python
# Text file
with open('file.txt', 'w') as f:
    f.write('content')

# CSV
with open('file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['col1', 'col2'])
    writer.writerow([1, 2])

# JSON
with open('file.json', 'w') as f:
    json.dump(data, f, indent=4)
```

---

## List & Dictionary Operations

### Lists

```python
# Creating
lst = [1, 2, 3]
lst = list(range(10))

# Adding
lst.append(4)
lst.extend([5, 6])
lst.insert(0, 0)

# Removing
lst.remove(3)
lst.pop()  # Remove last
lst.pop(0)  # Remove by index
del lst[0]

# Slicing
lst[0:3]      # First 3
lst[-3:]      # Last 3
lst[::2]      # Every 2nd element
lst[::-1]     # Reverse

# List comprehension
[x**2 for x in range(10)]
[x for x in lst if x > 5]
[x**2 if x > 5 else x for x in lst]
```

### Dictionaries

```python
# Creating
d = {'key1': 'value1', 'key2': 'value2'}
print(d)  # Output: {'key1': 'value1', 'key2': 'value2'}

d = dict(key1='value1', key2='value2')
print(d)  # Output: {'key1': 'value1', 'key2': 'value2'}

# Accessing
print(d['key1'])  # Output: value1
print(d.get('key1', 'default'))  # Output: value1
print(d.get('key3', 'default'))  # Output: default (key doesn't exist)

# Adding/Updating
d['key3'] = 'value3'
print(d)  # Output: {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

d.update({'key4': 'value4'})
print(d)  # Output: {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}

# Removing
del d['key1']
print(d)  # Output: {'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}

removed = d.pop('key2')
print(f"Removed: {removed}, Dict: {d}")  # Output: Removed: value2, Dict: {'key3': 'value3', 'key4': 'value4'}

last = d.popitem()  # Remove last item
print(f"Removed: {last}, Dict: {d}")  # Output: Removed: ('key4', 'value4'), Dict: {'key3': 'value3'}

# Iterating
for key, value in d.items():
    print(key, value)

for key in d.keys():
    print(key)

for value in d.values():
    print(value)

# Dictionary comprehension
{key: value**2 for key, value in d.items()}
```

---

## String Operations

```python
# Basic operations
s = "Hello World"
print(s.lower())        # Output: hello world
print(s.upper())        # Output: HELLO WORLD
print(s.strip())        # Output: Hello World (removes whitespace)

s_with_spaces = "  Hello World  "
print(s_with_spaces.strip())  # Output: Hello World

print(s.split(' '))     # Output: ['Hello', 'World']
print(s.replace('World', 'Python'))  # Output: Hello Python
print(s.startswith('H'))  # Output: True
print(s.endswith('d'))    # Output: True
print(s.find('World'))    # Output: 6 (index where 'World' starts)
print(s.count('l'))      # Output: 3 (count of 'l')

# Formatting
value = 42
print(f"Value: {value}")  # Output: Value: 42
print("Value: {}".format(value))  # Output: Value: 42
print("Value: {:.2f}".format(3.14159))  # Output: Value: 3.14

# Checking
print("123".isdigit())   # Output: True
print("abc".isalpha())   # Output: True
print("abc123".isalnum())  # Output: True
print("abc 123".isalnum())  # Output: False (contains space)
```

---

## Date & Time

```python
from datetime import datetime, timedelta

# Current time
now = datetime.now()
print(now)  # Output: 2024-01-15 14:30:45.123456

today = datetime.today()
print(today)  # Output: 2024-01-15 14:30:45.123456

# Creating dates
date = datetime(2024, 1, 1)
print(date)  # Output: 2024-01-01 00:00:00

date = datetime.strptime('2024-01-01', '%Y-%m-%d')
print(date)  # Output: 2024-01-01 00:00:00

# Formatting
formatted = date.strftime('%Y-%m-%d')
print(formatted)  # Output: 2024-01-01

formatted = date.strftime('%B %d, %Y')
print(formatted)  # Output: January 01, 2024

# Operations
date1 = datetime(2024, 1, 1)
date2 = date1 + timedelta(days=7)
print(date2)  # Output: 2024-01-08 00:00:00

date3 = date1 - timedelta(days=7)
print(date3)  # Output: 2023-12-25 00:00:00

date2 = datetime(2024, 1, 8)
diff = (date2 - date1).days
print(diff)  # Output: 7
```

### Pandas Date Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

# Filtering
df[df['date'] > '2024-01-01']
df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]
```

---

## Useful Functions

### Lambda Functions

```python
# Simple function
square = lambda x: x**2

# With map
squared = list(map(lambda x: x**2, [1, 2, 3]))

# With filter
evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))

# With pandas
df['new_col'] = df['col'].apply(lambda x: x**2)
```

### Zip and Enumerate

```python
# Zip
for x, y in zip(list1, list2):
    print(x, y)

# Enumerate
for i, value in enumerate(list1):
    print(i, value)
```

### Any and All

```python
any([True, False, True])  # True
all([True, True, True])   # True
```

### Sorting

```python
sorted(lst)
sorted(lst, reverse=True)
sorted(lst, key=lambda x: x[1])  # Sort by second element

# In-place
lst.sort()
lst.sort(reverse=True)
```

### Set Operations

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

set1.union(set2)      # {1, 2, 3, 4, 5}
set1.intersection(set2)  # {3}
set1.difference(set2)  # {1, 2}
```

---

## Quick Tips

1. **Always use virtual environments**: `python -m venv env`
2. **Import conventions**:
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```
3. **Set random seeds**: `np.random.seed(42)`, `random_state=42`
4. **Use `.copy()`** when modifying DataFrames to avoid SettingWithCopyWarning
5. **Check data types**: Always verify with `df.dtypes` and `df.info()`
6. **Handle missing data early**: Check with `df.isna().sum()`
7. **Save your work**: Use version control and save intermediate results

---

**Print this cheatsheet and keep it handy!** Practice these commands daily to commit them to memory.

