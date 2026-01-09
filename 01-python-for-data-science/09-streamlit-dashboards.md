# Streamlit for Interactive Dashboards

Complete guide to building interactive web applications and dashboards with Streamlit for data science and machine learning.

## Table of Contents

- [Introduction to Streamlit](#introduction-to-streamlit)
- [Installation and Setup](#installation-and-setup)
- [Basic Streamlit Components](#basic-streamlit-components)
- [Data Visualization](#data-visualization)
- [Interactive Widgets](#interactive-widgets)
- [Building ML Dashboards](#building-ml-dashboards)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Complete Example: ML Model Dashboard](#complete-example-ml-model-dashboard)

---

## Introduction to Streamlit

### What is Streamlit?

Streamlit is an open-source Python framework for building interactive web applications for data science and machine learning. It allows you to create beautiful dashboards and apps with minimal code.

**Key Features:**
- **Simple**: Write Python code, get a web app
- **Fast**: Rapid prototyping and deployment
- **Interactive**: Built-in widgets and components
- **No Frontend Required**: No HTML, CSS, or JavaScript needed
- **Pythonic**: Works seamlessly with pandas, matplotlib, plotly, etc.

### Why Use Streamlit?

**Advantages:**
- Quick prototyping of data apps
- Share insights with non-technical stakeholders
- Deploy ML models as interactive apps
- Create data exploration tools
- Build internal dashboards

**Use Cases:**
- Data exploration dashboards
- ML model demos and interfaces
- Data analysis reports
- Interactive visualizations
- Internal tools and utilities

---

## Installation and Setup

### Installation

```bash
pip install streamlit
```

### Running Your First App

Create a file `app.py`:

```python
import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, World!")
```

Run the app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Basic Streamlit Components

### Text Elements

```python
import streamlit as st

# Title and headers
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.text("Regular text")

# Markdown
st.markdown("# Markdown Title")
st.markdown("**Bold text** and *italic text*")
st.markdown("[Link](https://streamlit.io)")

# Code blocks
st.code("""
def hello():
    print("Hello, Streamlit!")
""", language='python')

# LaTeX
st.latex(r"E = mc^2")
```

### Data Display

```python
import pandas as pd
import numpy as np

# DataFrames
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100)
})
st.dataframe(df)
st.dataframe(df, height=300)  # With height

# Tables (static)
st.table(df.head(10))

# JSON
data = {'key': 'value', 'number': 42}
st.json(data)

# Metrics
st.metric("Temperature", "70 °F", "2 °F")
st.metric("Sales", "$1,234", "-5%")
```

### Images and Media

```python
from PIL import Image

# Images
image = Image.open("image.jpg")
st.image(image, caption="My Image", use_column_width=True)

# Videos
st.video("video.mp4")

# Audio
st.audio("audio.mp3")
```

---

## Data Visualization

### Matplotlib and Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
st.pyplot(fig)

# Seaborn
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='A', y='B', ax=ax)
st.pyplot(fig)
```

### Plotly (Interactive)

```python
import plotly.express as px

# Interactive scatter plot
fig = px.scatter(df, x='A', y='B', color='category')
st.plotly_chart(fig)

# Interactive bar chart
fig = px.bar(df, x='category', y='value')
st.plotly_chart(fig, use_container_width=True)
```

### Built-in Charts

```python
# Line chart
st.line_chart(df)

# Area chart
st.area_chart(df)

# Bar chart
st.bar_chart(df)

# Map
df_map = pd.DataFrame({
    'lat': [37.76, 37.77, 37.78],
    'lon': [-122.4, -122.41, -122.42]
})
st.map(df_map)
```

---

## Interactive Widgets

### Input Widgets

```python
# Text input
name = st.text_input("Enter your name", "Default value")
st.write(f"Hello, {name}!")

# Number input
age = st.number_input("Enter your age", min_value=0, max_value=120, value=25)

# Text area
text = st.text_area("Enter text", height=200)

# Date input
date = st.date_input("Select a date")

# Time input
time = st.time_input("Select a time")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
```

### Selection Widgets

```python
# Selectbox
option = st.selectbox("Choose an option", ['Option 1', 'Option 2', 'Option 3'])

# Multiselect
options = st.multiselect("Choose options", ['A', 'B', 'C', 'D'])

# Radio buttons
choice = st.radio("Choose one", ['Option 1', 'Option 2'])

# Checkbox
agree = st.checkbox("I agree to the terms")
if agree:
    st.write("Great!")
```

### Sliders

```python
# Slider
value = st.slider("Select a value", 0, 100, 50)

# Range slider
values = st.slider("Select a range", 0, 100, (25, 75))

# Float slider
float_value = st.slider("Float value", 0.0, 1.0, 0.5)
```

### Buttons

```python
# Button
if st.button("Click me"):
    st.write("Button clicked!")

# Download button
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)
st.download_button("Download CSV", csv, "data.csv", "text/csv")
```

---

## Building ML Dashboards

### Model Prediction Interface

```python
import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("ML Model Predictor")

# Input features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Make prediction
if st.button("Predict"):
    input_data = [[feature1, feature2, feature3]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.success(f"Prediction: {prediction}")
    st.write(f"Probability: {probability}")
```

### Data Exploration Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Data Exploration Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    columns = st.sidebar.multiselect("Select columns", df.columns.tolist())
    
    if columns:
        # Display filtered data
        st.dataframe(df[columns])
        
        # Visualizations
        chart_type = st.selectbox("Chart type", ['Scatter', 'Bar', 'Line', 'Histogram'])
        
        if chart_type == 'Scatter' and len(columns) >= 2:
            fig = px.scatter(df, x=columns[0], y=columns[1])
            st.plotly_chart(fig)
        
        elif chart_type == 'Bar' and len(columns) >= 1:
            fig = px.bar(df, x=columns[0])
            st.plotly_chart(fig)
        
        elif chart_type == 'Line' and len(columns) >= 2:
            fig = px.line(df, x=columns[0], y=columns[1])
            st.plotly_chart(fig)
        
        elif chart_type == 'Histogram' and len(columns) >= 1:
            fig = px.histogram(df, x=columns[0])
            st.plotly_chart(fig)
```

### Model Performance Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

st.title("Model Performance Dashboard")

# Load predictions
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 1])
y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3, 0.7, 0.6, 0.9, 0.8])

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{np.mean(y_true == y_pred):.2%}")
with col2:
    st.metric("Precision", "0.83")
with col3:
    st.metric("Recall", "0.83")
with col4:
    st.metric("F1-Score", "0.83")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig = px.imshow(cm, text_auto=True, aspect="auto", 
                labels=dict(x="Predicted", y="Actual"),
                x=['Class 0', 'Class 1'],
                y=['Class 0', 'Class 1'])
st.plotly_chart(fig)

# ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

fig = px.line(x=fpr, y=tpr, 
              title=f'ROC Curve (AUC = {roc_auc:.2f})',
              labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
fig.add_shape(type='line', line=dict(dash='dash'),
              x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(fig)
```

---

## Advanced Features

### Caching

```python
@st.cache
def expensive_computation(data):
    # This function will only run once and cache the result
    return data * 2

result = expensive_computation(large_data)
```

### Session State

```python
# Initialize session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Increment counter
if st.button("Increment"):
    st.session_state.counter += 1

st.write(f"Counter: {st.session_state.counter}")
```

### Sidebar

```python
# Everything in sidebar
st.sidebar.title("Sidebar Title")
st.sidebar.selectbox("Choose", ['A', 'B', 'C'])
```

### Columns

```python
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Column 1")
    
with col2:
    st.write("Column 2")
    
with col3:
    st.write("Column 3")
```

### Expander

```python
with st.expander("Click to expand"):
    st.write("Hidden content here")
```

---

## Deployment

### Streamlit Cloud (Free)

1. Push your app to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Other Platforms

- **Heroku**: Use `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **AWS/Azure/GCP**: Deploy as containerized app
- **Docker**: Create Dockerfile for containerization

---

## Best Practices

1. **Use caching**: `@st.cache` for expensive operations
2. **Organize with sidebar**: Put controls in sidebar
3. **Use columns**: Organize layout with `st.columns()`
4. **Error handling**: Use try/except for robust apps
5. **Documentation**: Add markdown explanations
6. **Performance**: Optimize data loading and processing

---

## Complete Example: ML Model Dashboard

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("Machine Learning Model Dashboard")

# Sidebar
st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)

# Load data
@st.cache
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

df = load_data()

# Display data
st.subheader("Dataset")
st.dataframe(df.head())

# Train model
if st.button("Train Model"):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.success(f"Model Accuracy: {accuracy:.2%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='feature', y='importance')
    st.plotly_chart(fig)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# Prediction interface
st.sidebar.header("Make Prediction")
sepal_length = st.sidebar.number_input("Sepal Length", value=5.0)
sepal_width = st.sidebar.number_input("Sepal Width", value=3.0)
petal_length = st.sidebar.number_input("Petal Length", value=1.5)
petal_width = st.sidebar.number_input("Petal Width", value=0.5)

if st.sidebar.button("Predict"):
    # Load or use cached model
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)
    
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.sidebar.success(f"Prediction: Class {prediction}")
    st.sidebar.write(f"Probabilities: {probability}")
```

---

## Key Takeaways

1. **Streamlit is simple**: Build apps with just Python
2. **Interactive widgets**: Create user-friendly interfaces
3. **Visualization**: Integrate matplotlib, plotly, seaborn
4. **Caching**: Use `@st.cache` for performance
5. **Deploy easily**: Streamlit Cloud for free hosting
6. **Great for ML**: Perfect for model demos and dashboards

---

**Remember**: Streamlit makes it easy to share your data science work with others. Start simple and add complexity as needed!

