# Python Data Science Cheatsheet

Quick reference for common Python syntax and operations used in everyday data science work.

## Table of Contents

- [NumPy](#numpy)
- [Pandas](#pandas)
- [Matplotlib & Seaborn](#matplotlib--seaborn)
- [Scikit-learn](#scikit-learn)
- [PyTorch](#pytorch)
- [TensorFlow/Keras](#tensorflowkeras)
- [OpenCV](#opencv)
- [Hyperparameter Tuning & Model Optimization](#hyperparameter-tuning--model-optimization)
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
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
empty = np.empty((2, 2))
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
random = np.random.rand(3, 3)  # Random 0-1
randint = np.random.randint(0, 10, (3, 3))  # Random integers
```

### Array Operations

```python
# Shape and dimensions
arr.shape
arr.ndim
arr.size
arr.dtype

# Reshaping
arr.reshape(2, 3)
arr.flatten()
arr.ravel()

# Indexing and slicing
arr[0]           # First element
arr[-1]          # Last element
arr[1:4]         # Slice
arr[arr > 5]     # Boolean indexing
arr2d[0, 1]      # 2D indexing
arr2d[:, 1]      # All rows, column 1
arr2d[1, :]      # Row 1, all columns

# Mathematical operations
np.sum(arr)
np.mean(arr)
np.std(arr)
np.var(arr)
np.min(arr)
np.max(arr)
np.median(arr)
np.percentile(arr, 50)

# Element-wise operations
arr + 1
arr * 2
arr ** 2
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.sin(arr)

# Array operations
arr1 + arr2      # Element-wise addition
arr1 * arr2      # Element-wise multiplication
np.dot(arr1, arr2)  # Matrix multiplication
arr1 @ arr2      # Matrix multiplication (Python 3.5+)
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
df.head()        # First 5 rows
df.head(10)      # First 10 rows
df.tail()        # Last 5 rows
df.shape         # (rows, columns)
df.info()        # Data types and memory
df.describe()    # Statistical summary
df.columns       # Column names
df.index         # Row indices
df.dtypes        # Data types
df.value_counts()  # Value counts for Series
```

### Selecting Data

```python
# Single column (returns Series)
df['column_name']
df.column_name

# Multiple columns (returns DataFrame)
df[['col1', 'col2']]

# Rows by index
df.iloc[0]           # First row
df.iloc[0:5]        # First 5 rows
df.iloc[0, 1]       # Row 0, column 1
df.iloc[0:5, 1:3]   # Rows 0-4, columns 1-2

# Rows by label
df.loc['index_name']
df.loc['row1':'row5', 'col1':'col3']

# Boolean indexing
df[df['column'] > 5]
df[(df['col1'] > 5) & (df['col2'] < 10)]
df[df['column'].isin([1, 2, 3])]
df[df['column'].str.contains('text')]
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
df.groupby('column').mean()
df.groupby('column').sum()
df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})

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
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities[:, 1])

# Regression
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
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
d = dict(key1='value1', key2='value2')

# Accessing
d['key1']
d.get('key1', 'default')  # With default

# Adding/Updating
d['key3'] = 'value3'
d.update({'key4': 'value4'})

# Removing
del d['key1']
d.pop('key2')
d.popitem()  # Remove last item

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
s.lower()
s.upper()
s.strip()
s.split(' ')
s.replace('old', 'new')
s.startswith('H')
s.endswith('d')
s.find('World')
s.count('l')

# Formatting
f"Value: {value}"
"Value: {}".format(value)
"Value: {:.2f}".format(3.14159)

# Checking
s.isdigit()
s.isalpha()
s.isalnum()
```

---

## Date & Time

```python
from datetime import datetime, timedelta

# Current time
now = datetime.now()
today = datetime.today()

# Creating dates
date = datetime(2024, 1, 1)
date = datetime.strptime('2024-01-01', '%Y-%m-%d')

# Formatting
date.strftime('%Y-%m-%d')
date.strftime('%B %d, %Y')

# Operations
date + timedelta(days=7)
date - timedelta(days=7)
(date2 - date1).days
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

