# Complete Data Science Project Tutorial

Step-by-step walkthrough of a complete end-to-end data science project from data collection to interactive dashboard.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Collection (Web Scraping)](#step-1-data-collection-web-scraping)
- [Step 2: Data Cleaning and Preprocessing](#step-2-data-cleaning-and-preprocessing)
- [Step 3: Exploratory Data Analysis](#step-3-exploratory-data-analysis)
- [Step 4: Data Visualization](#step-4-data-visualization)
- [Step 5: Building Interactive Dashboard](#step-5-building-interactive-dashboard)
- [Step 6: Insights and Conclusions](#step-6-insights-and-conclusions)

---

## Project Overview

**Project**: Analyze Real Estate Prices

**Goal**: Collect, analyze, and visualize real estate data to understand market trends

**Tools Used**:
- NumPy: Numerical operations
- Pandas: Data manipulation
- Matplotlib/Seaborn: Static visualizations
- Plotly: Interactive visualizations
- Streamlit: Interactive dashboard
- BeautifulSoup/Selenium: Web scraping

**Time**: 3-4 hours

---

## Step 1: Data Collection (Web Scraping)

### Option A: Using Public API

```python
import requests
import pandas as pd
import json

def fetch_real_estate_data(api_url, params):
    """Fetch data from API"""
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Error: {response.status_code}")
        return None

# Example: Using a sample API
# In practice, use actual real estate API
api_url = "https://api.example.com/real-estate"
params = {"city": "New York", "limit": 1000}

df = fetch_real_estate_data(api_url, params)
```

### Option B: Web Scraping with BeautifulSoup

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

def scrape_real_estate_data(url, max_pages=10):
    """Scrape real estate data from website"""
    all_data = []
    
    for page in range(1, max_pages + 1):
        page_url = f"{url}?page={page}"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data (adjust selectors based on website)
        listings = soup.find_all('div', class_='listing')
        
        for listing in listings:
            data = {
                'price': extract_price(listing),
                'bedrooms': extract_bedrooms(listing),
                'bathrooms': extract_bathrooms(listing),
                'sqft': extract_sqft(listing),
                'location': extract_location(listing),
                'date': extract_date(listing)
            }
            all_data.append(data)
        
        time.sleep(1)  # Be respectful
    
    return pd.DataFrame(all_data)

# Helper functions
def extract_price(listing):
    # Implementation
    pass

# For this tutorial, we'll use synthetic data
import numpy as np

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'price': np.random.normal(500000, 150000, n_samples),
    'bedrooms': np.random.randint(1, 5, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft': np.random.normal(2000, 500, n_samples),
    'location': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
    'year_built': np.random.randint(1950, 2023, n_samples),
    'date_listed': pd.date_range('2023-01-01', periods=n_samples, freq='D')
})

# Ensure positive values
df['price'] = np.abs(df['price'])
df['sqft'] = np.abs(df['sqft'])

print("Data collected!")
print(f"Shape: {df.shape}")
print(df.head())
```

---

## Step 2: Data Cleaning and Preprocessing

### Load and Inspect Data

```python
# Basic info
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
```

### Handle Missing Values

```python
# Check for missing values
missing_pct = df.isnull().sum() / len(df) * 100
print("Missing Values Percentage:")
print(missing_pct)

# Handle missing values
# For numerical: fill with median
df['price'] = df['price'].fillna(df['price'].median())
df['sqft'] = df['sqft'].fillna(df['sqft'].median())

# For categorical: fill with mode
df['location'] = df['location'].fillna(df['location'].mode()[0])

# Or drop rows with missing values
df = df.dropna()
```

### Handle Outliers

```python
# Detect outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from price and sqft
df_clean = remove_outliers_iqr(df, 'price')
df_clean = remove_outliers_iqr(df_clean, 'sqft')

print(f"Original shape: {df.shape}")
print(f"After removing outliers: {df_clean.shape}")
```

### Feature Engineering

```python
# Create new features
df_clean['price_per_sqft'] = df_clean['price'] / df_clean['sqft']
df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
df_clean['age'] = 2023 - df_clean['year_built']
df_clean['month_listed'] = df_clean['date_listed'].dt.month
df_clean['year_listed'] = df_clean['date_listed'].dt.year

# Categorical encoding
df_clean = pd.get_dummies(df_clean, columns=['location'], prefix='loc')

print("New features created!")
print(df_clean.columns.tolist())
```

---

## Step 3: Exploratory Data Analysis

### Univariate Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Distribution of price
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price distribution
axes[0, 0].hist(df_clean['price'], bins=50, edgecolor='black')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')

# SQFT distribution
axes[0, 1].hist(df_clean['sqft'], bins=50, edgecolor='black', color='orange')
axes[0, 1].set_title('Square Footage Distribution')
axes[0, 1].set_xlabel('Square Feet')
axes[0, 1].set_ylabel('Frequency')

# Bedrooms distribution
df_clean['bedrooms'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Bedrooms Distribution')
axes[1, 0].set_xlabel('Number of Bedrooms')
axes[1, 0].set_ylabel('Count')

# Location distribution
location_cols = [col for col in df_clean.columns if col.startswith('loc_')]
location_counts = df_clean[location_cols].sum()
location_counts.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Location Distribution')
axes[1, 1].set_xlabel('Location')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Bivariate Analysis

```python
# Correlation analysis
numeric_cols = ['price', 'bedrooms', 'bathrooms', 'sqft', 'price_per_sqft', 'age']
correlation_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Price vs SQFT
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(df_clean['sqft'], df_clean['price'], alpha=0.5)
axes[0].set_xlabel('Square Feet')
axes[0].set_ylabel('Price ($)')
axes[0].set_title('Price vs Square Footage')
axes[0].grid(True, alpha=0.3)

# Price by Bedrooms
df_clean.boxplot(column='price', by='bedrooms', ax=axes[1])
axes[1].set_xlabel('Number of Bedrooms')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Price Distribution by Bedrooms')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Statistical Summary

```python
# Group by location
location_stats = df_clean.groupby('location').agg({
    'price': ['mean', 'median', 'std'],
    'sqft': ['mean', 'median'],
    'bedrooms': 'mean'
}).round(2)

print("Statistics by Location:")
print(location_stats)

# Group by bedrooms
bedroom_stats = df_clean.groupby('bedrooms').agg({
    'price': ['mean', 'count'],
    'sqft': 'mean',
    'price_per_sqft': 'mean'
}).round(2)

print("\nStatistics by Bedrooms:")
print(bedroom_stats)
```

---

## Step 4: Data Visualization

### Static Visualizations with Matplotlib/Seaborn

```python
# Advanced visualization
fig = plt.figure(figsize=(16, 10))

# Price trends over time
ax1 = plt.subplot(2, 3, 1)
monthly_avg_price = df_clean.groupby('month_listed')['price'].mean()
monthly_avg_price.plot(kind='line', marker='o', ax=ax1)
ax1.set_title('Average Price by Month')
ax1.set_xlabel('Month')
ax1.set_ylabel('Average Price ($)')
ax1.grid(True, alpha=0.3)

# Price distribution by location
ax2 = plt.subplot(2, 3, 2)
df_clean.boxplot(column='price', by='location', ax=ax2)
ax2.set_title('Price Distribution by Location')
ax2.set_xlabel('Location')
ax2.set_ylabel('Price ($)')

# Price per SQFT by bedrooms
ax3 = plt.subplot(2, 3, 3)
df_clean.boxplot(column='price_per_sqft', by='bedrooms', ax=ax3)
ax3.set_title('Price per SQFT by Bedrooms')
ax3.set_xlabel('Bedrooms')
ax3.set_ylabel('Price per SQFT ($)')

# Age vs Price
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(df_clean['age'], df_clean['price'], alpha=0.5)
ax4.set_xlabel('Age (years)')
ax4.set_ylabel('Price ($)')
ax4.set_title('Price vs Age')
ax4.grid(True, alpha=0.3)

# Bedrooms vs Bathrooms heatmap
ax5 = plt.subplot(2, 3, 5)
bed_bath_counts = pd.crosstab(df_clean['bedrooms'], df_clean['bathrooms'])
sns.heatmap(bed_bath_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax5)
ax5.set_title('Bedrooms vs Bathrooms')
ax5.set_xlabel('Bathrooms')
ax5.set_ylabel('Bedrooms')

# Price distribution
ax6 = plt.subplot(2, 3, 6)
df_clean['price'].hist(bins=50, edgecolor='black', ax=ax6)
ax6.axvline(df_clean['price'].mean(), color='r', linestyle='--', label='Mean')
ax6.axvline(df_clean['price'].median(), color='g', linestyle='--', label='Median')
ax6.set_xlabel('Price ($)')
ax6.set_ylabel('Frequency')
ax6.set_title('Price Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('real_estate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Interactive Visualizations with Plotly

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Interactive scatter plot
fig = px.scatter(df_clean, x='sqft', y='price', 
                 color='bedrooms', size='bathrooms',
                 hover_data=['location', 'age'],
                 title='Interactive Price vs Square Footage')
fig.update_layout(width=1000, height=600)
fig.show()

# Interactive time series
monthly_data = df_clean.groupby('month_listed').agg({
    'price': 'mean',
    'sqft': 'mean',
    'bedrooms': 'mean'
}).reset_index()

fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('Average Price by Month', 'Average SQFT by Month'))

fig.add_trace(
    go.Scatter(x=monthly_data['month_listed'], y=monthly_data['price'],
               mode='lines+markers', name='Price'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=monthly_data['month_listed'], y=monthly_data['sqft'],
               mode='lines+markers', name='SQFT'),
    row=2, col=1
)

fig.update_xaxes(title_text="Month", row=2, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Square Feet", row=2, col=1)
fig.update_layout(height=800, title_text="Monthly Trends")
fig.show()
```

---

## Step 5: Building Interactive Dashboard

### Streamlit Dashboard

```python
# Create app.py for Streamlit
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Analysis", layout="wide")

# Title
st.title("🏠 Real Estate Market Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    return df_clean

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

location_filter = st.sidebar.multiselect(
    "Select Locations",
    options=df['location'].unique(),
    default=df['location'].unique()
)

bedroom_filter = st.sidebar.multiselect(
    "Select Bedrooms",
    options=sorted(df['bedrooms'].unique()),
    default=sorted(df['bedrooms'].unique())
)

price_range = st.sidebar.slider(
    "Price Range",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max()))
)

# Filter data
filtered_df = df[
    (df['location'].isin(location_filter)) &
    (df['bedrooms'].isin(bedroom_filter)) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1])
]

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Listings", len(filtered_df))
with col2:
    st.metric("Average Price", f"${filtered_df['price'].mean():,.0f}")
with col3:
    st.metric("Average SQFT", f"{filtered_df['sqft'].mean():,.0f}")
with col4:
    st.metric("Price per SQFT", f"${filtered_df['price_per_sqft'].mean():,.0f}")

# Charts
col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(filtered_df, x='sqft', y='price', 
                     color='bedrooms', size='bathrooms',
                     title='Price vs Square Footage')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(filtered_df, x='location', y='price',
                 title='Price Distribution by Location')
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader("Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Run with: streamlit run app.py
```

---

## Step 6: Insights and Conclusions

### Key Insights

```python
# Generate insights
insights = []

# Insight 1: Price trends
avg_price = df_clean['price'].mean()
insights.append(f"Average property price: ${avg_price:,.0f}")

# Insight 2: Location impact
location_impact = df_clean.groupby('location')['price'].mean().sort_values(ascending=False)
insights.append(f"Most expensive location: {location_impact.index[0]} (${location_impact.iloc[0]:,.0f})")
insights.append(f"Least expensive location: {location_impact.index[-1]} (${location_impact.iloc[-1]:,.0f})")

# Insight 3: Bedroom impact
bedroom_impact = df_clean.groupby('bedrooms')['price'].mean()
insights.append(f"Price increases by ${bedroom_impact.diff().mean():,.0f} per additional bedroom on average")

# Insight 4: Price per SQFT
avg_price_per_sqft = df_clean['price_per_sqft'].mean()
insights.append(f"Average price per square foot: ${avg_price_per_sqft:,.0f}")

# Insight 5: Age impact
correlation_age_price = df_clean['age'].corr(df_clean['price'])
insights.append(f"Age-Price correlation: {correlation_age_price:.3f}")

print("Key Insights:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")
```

### Export Results

```python
# Save cleaned data
df_clean.to_csv('real_estate_cleaned.csv', index=False)

# Save summary statistics
summary_stats = df_clean.describe()
summary_stats.to_csv('summary_statistics.csv')

# Save insights
with open('insights.txt', 'w') as f:
    f.write("Real Estate Analysis Insights\n")
    f.write("=" * 40 + "\n\n")
    for insight in insights:
        f.write(f"{insight}\n")

print("Results exported successfully!")
```

---

## Complete Code Summary

```python
# Complete Data Science Project Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# 1. Data Collection
df = collect_data()  # Web scraping or API

# 2. Data Cleaning
df_clean = clean_data(df)

# 3. Feature Engineering
df_clean = engineer_features(df_clean)

# 4. EDA
perform_eda(df_clean)

# 5. Visualization
create_visualizations(df_clean)

# 6. Dashboard
create_dashboard(df_clean)

# 7. Insights
generate_insights(df_clean)
```

---

## Key Takeaways

1. **End-to-End Process**: You've completed a full data science project
2. **Tool Integration**: Combined NumPy, Pandas, Matplotlib, Plotly, Streamlit
3. **Best Practices**: Proper data cleaning, EDA, and visualization
4. **Real-World Skills**: Web scraping, data analysis, dashboard creation

---

**Congratulations!** You've completed a comprehensive data science project using all the tools you've learned!

