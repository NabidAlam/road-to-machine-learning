# Project 6: Customer Data Dashboard with Streamlit

Build an interactive dashboard to visualize and analyze customer data using Streamlit.

## Difficulty
Beginner

## Time Estimate
2-3 days

## Skills You'll Practice
- Streamlit
- Data Visualization
- Interactive Dashboards
- Data Analysis
- User Interface Design

## Learning Objectives

By completing this project, you will learn to:
- Build interactive web applications with Streamlit
- Create dynamic visualizations
- Implement user input and filtering
- Design user-friendly dashboards
- Deploy Streamlit applications
- Work with real-world customer data

## Prerequisites

Before starting, you should have completed:
- Phase 0: Prerequisites (Python Basics)
- Phase 1: Python for Data Science (Pandas, Matplotlib, Seaborn)
- Basic understanding of data visualization

## Dataset

**Option 1: E-commerce Customer Dataset**
- [E-commerce Customer Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- Contains customer demographics, purchase history, and behavior

**Option 2: Customer Churn Dataset**
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Customer information and churn status

**Option 3: Create Your Own**
- Use any customer-related dataset from Kaggle or UCI ML Repository

## Project Steps

### Step 1: Setup and Data Loading
- Install Streamlit: `pip install streamlit`
- Load customer dataset
- Explore data structure
- Clean and preprocess data

### Step 2: Basic Dashboard Structure
- Create main Streamlit app file (`app.py`)
- Set up page configuration
- Add title and description
- Create sidebar for navigation

### Step 3: Data Overview Section
- Display dataset summary
- Show basic statistics
- Display data sample
- Add data shape information

### Step 4: Interactive Visualizations
- Create charts with user controls:
  - Bar charts for categorical data
  - Line charts for trends
  - Scatter plots for relationships
  - Histograms for distributions
- Add filters and dropdowns
- Implement date range selectors

### Step 5: Analysis Sections
- Customer segmentation visualization
- Purchase behavior analysis
- Geographic distribution (if available)
- Time-based trends

### Step 6: Advanced Features
- Add download functionality for filtered data
- Implement search/filter capabilities
- Create comparison views
- Add insights and summary statistics

### Step 7: Deployment (Optional)
- Deploy to Streamlit Cloud
- Or deploy to Heroku/AWS
- Share your dashboard

## Code Structure

```
project-06-customer-dashboard/
├── README.md
├── app.py                    # Main Streamlit application
├── data/
│   └── customer_data.csv     # Dataset
├── notebooks/
│   └── data_exploration.ipynb
├── requirements.txt
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Example Code Skeleton

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/customer_data.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.title("Filters")
# Add filters here

# Main content
st.title("Customer Data Dashboard")
st.markdown("Interactive dashboard for customer analysis")

# Overview section
st.header("Data Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
# Add more metrics

# Visualizations
st.header("Customer Analysis")
# Add charts here

# Run with: streamlit run app.py
```

## Key Streamlit Components to Use

- `st.title()`, `st.header()`, `st.subheader()` - Text elements
- `st.dataframe()`, `st.table()` - Display data
- `st.plotly_chart()`, `st.pyplot()` - Visualizations
- `st.selectbox()`, `st.slider()`, `st.multiselect()` - User inputs
- `st.columns()` - Layout
- `st.sidebar` - Sidebar elements
- `st.cache_data` - Data caching

## Extensions

1. **Add Machine Learning Predictions**
   - Integrate a simple classification model
   - Show predictions in the dashboard
   - Add model performance metrics

2. **Real-time Data Updates**
   - Connect to a database
   - Auto-refresh functionality
   - Live data updates

3. **Multi-page Dashboard**
   - Use `st.sidebar` for navigation
   - Create separate pages for different analyses
   - Use `st.session_state` for state management

4. **Advanced Visualizations**
   - Interactive Plotly charts
   - Maps with `st.map()` or `folium`
   - 3D visualizations

5. **Export Functionality**
   - Download filtered data as CSV
   - Export charts as images
   - Generate PDF reports

## Evaluation Criteria

Your dashboard should:
- ✅ Load and display data correctly
- ✅ Have clear, intuitive navigation
- ✅ Include multiple interactive visualizations
- ✅ Allow users to filter and explore data
- ✅ Be visually appealing and well-organized
- ✅ Include meaningful insights
- ✅ Have proper error handling

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)
- [Plotly with Streamlit](https://plotly.com/python/streamlit/)

## Tips for Success

1. **Start Simple**: Begin with basic visualizations, then add interactivity
2. **Use Caching**: Use `@st.cache_data` for data loading to improve performance
3. **Organize Well**: Use columns and containers for better layout
4. **Test Interactively**: Run `streamlit run app.py` and test as you build
5. **Keep It Fast**: Optimize data loading and processing
6. **Make It Beautiful**: Use Streamlit's built-in styling options

## Next Steps

After completing this project:
- Try deploying your dashboard
- Add more advanced features
- Build dashboards for other datasets
- Move to intermediate projects that use Streamlit for ML applications

---

**Ready to build?** Start by setting up your environment and loading your dataset!

