# Module 01: Python for Data Science

Master the essential Python libraries for data manipulation, analysis, and visualization: your **data processing & visualization (ML toolbox)** for everything that follows.

##  What You'll Learn

- NumPy for numerical computing (`ndarray`, dtypes, broadcasting, linear algebra)
- Pandas for data manipulation (Series, DataFrame, I/O, `groupby`, missing data, datetimes)
- Matplotlib and Seaborn for data visualization (line, scatter, histogram, bar, pie, and beyond)
- Plotly and Dash for interactive visualizations
- Streamlit for building dashboards and ML applications
- Flask for web applications and REST APIs
- Tableau for professional data visualization
- Exploratory Data Analysis (EDA) techniques
- Working with APIs, databases, and web scraping

## ML toolbox curriculum map

Each line below is covered **with code examples** in the linked guide (same style as the Python basics doc: concept + runnable snippets). The **from-scratch ML project** uses tutorials that combine these tools.

| Topic | Where to study |
|-------|----------------|
| **Data processing & visualization (ML toolbox)** | This phase as a whole; start with [NumPy](01-numpy.md#ml-toolbox-curriculum-map-this-guide) → [Pandas](02-pandas.md#ml-toolbox-curriculum-map-this-guide) → [Visualization](03-visualization.md#ml-toolbox-curriculum-map-this-guide) |
| NumPy fundamentals, `ndarray`, attributes & dtypes | [NumPy → Attributes](01-numpy.md#array-attributes-and-methods), [Creating arrays](01-numpy.md#creating-arrays) |
| Pandas Series, DataFrame & file handling (CSV/Excel etc.) | [Pandas → Series/DataFrames](02-pandas.md#series-and-dataframes), [Reading/Writing](02-pandas.md#reading-and-writing-data) |
| Array creation (existing data, scratch, range, random) | [NumPy → Creating arrays](01-numpy.md#creating-arrays) |
| Data access: indexing, `loc` / `iloc` & filtering | [NumPy → Indexing](01-numpy.md#indexing-and-slicing), [Pandas → Selection](02-pandas.md#data-selection-and-filtering) |
| Indexing, slicing, copying & advanced iteration | [NumPy → Copies](01-numpy.md#deep-and-shallow-copy), [Pandas patterns](02-pandas.md#data-selection-and-filtering) |
| Modifying data (add/drop rows & columns, assign values) | [Pandas → cleaning/ops](02-pandas.md#data-cleaning) (drops, assigns); see also DataFrame basics |
| Array manipulation, reshaping & transformations | [NumPy → Reshape](01-numpy.md#reshaping-and-resizing), [Advanced manipulation](01-numpy.md#advanced-array-manipulation) |
| Duplicates, missing data & DateTime | [Pandas → Data cleaning](02-pandas.md#data-cleaning), [Time series / datetime](02-pandas.md#time-series-operations) |
| Arithmetic, mathematical & logical operations | [NumPy → Array operations](01-numpy.md#array-operations), [Mathematical ops](01-numpy.md#mathematical-operations) |
| Apply, aggregation & GroupBy | [Pandas → Grouping](02-pandas.md#grouping-and-aggregation), [apply / map](02-pandas.md#advanced-tricks-and-performance) |
| Broadcasting, sorting, searching & counting | [Broadcasting](01-numpy.md#broadcasting), [Sorting & searching](01-numpy.md#sorting-searching-and-counting) |
| Matplotlib line, scatter, histogram, bar & pie | [Visualization → Plot types](03-visualization.md#common-plot-types) |
| Statistical analysis & linear algebra basics | [NumPy stats](01-numpy.md#mathematical-operations), [Linear algebra](01-numpy.md#linear-algebra-operations) |
| **Project: Build an ML model from scratch** | [NumPy NN tutorial](../00-prerequisites/prerequisites-project-tutorial.md), [First ML project](../02-introduction-to-ml/first-ml-project-tutorial.md) |
| **Next: Machine learning for beginners** | [Intro ML curriculum map](../02-introduction-to-ml/introduction-to-ml.md#ml-for-beginners-curriculum-map-this-guide); EDA in [04-exploratory-data-analysis](04-exploratory-data-analysis.md#ml-for-beginners-curriculum-map-this-guide) |

##  Modules

### 01-numpy
Learn NumPy. The foundation of numerical computing in Python.

**Topics:**
- Fundamentals: `ndarray`, attributes, and data types
- Array creation: from existing data, from scratch, ranges, and random values
- Data access: indexing, slicing, views vs. copies, advanced iteration
- Reshaping and array transformations
- Arithmetic, mathematical, and logical operations
- Broadcasting; sorting, searching, and counting
- Statistical analysis and linear algebra basics

**Time Estimate:** 1 week

**[Complete Guide →](01-numpy.md)**

### 02-pandas
Master Pandas. The most important library for data manipulation.

**Topics:**
- Series and DataFrame; file handling (CSV, Excel, JSON, and related formats)
- Data access: indexing, `loc` / `iloc`, and filtering
- Indexing, slicing, copying, and advanced iteration patterns
- Modifying data: add/drop rows and columns, assign values
- Handling duplicates, missing data, and datetime operations
- Apply functions, aggregation, and `GroupBy` analysis
- Merging, joining, and concatenating DataFrames
- MultiIndex; stack/unstack, `melt`, pivot tables; vectorized string and datetime operations
- Time series operations

**Time Estimate:** 2 weeks

**[Complete Guide →](02-pandas.md)**

### 03-visualization
Create beautiful and informative visualizations.

**Topics:**
- Matplotlib: line, scatter, histogram, bar, and pie plots
- Seaborn statistical visualizations
- Plotly & Dash for Interactive Visualizations
- Customizing Plots
- Subplots and Multiple Plots
- Building Interactive Dashboards
- Saving Figures

**Time Estimate:** 1-2 weeks

**[Complete Guide →](03-visualization.md)**

### 04-exploratory-data-analysis
Systematic approach to understanding your data before modeling.

**Topics:**
- EDA Workflow and Best Practices
- Univariate Analysis (Numerical and Categorical)
- Bivariate Analysis (Relationships between variables)
- Multivariate Analysis (Correlations, PCA)
- Data Quality Checks (Outliers, Consistency)
- Creating EDA Reports

**Time Estimate:** 1-2 weeks

**[Complete Guide →](04-exploratory-data-analysis.md)**

### 05-data-sources-and-integration
Work with various data sources: APIs, databases, web scraping, and file formats.

**Topics:**
- Working with REST APIs
- Database Integration (SQL, NoSQL)
- Web Scraping (BeautifulSoup, Selenium)
- File Formats (CSV, Excel, JSON, Parquet, XML)
- Data Integration Pipelines
- Error Handling and Validation

**Time Estimate:** 1-2 weeks

**[Complete Guide →](05-data-sources-and-integration.md)**

### 06-regular-expressions-text-processing
Master regular expressions and text processing for cleaning and analyzing text data.

**Topics:**
- Regular Expressions Basics and Patterns
- Text Processing with Pandas
- Pattern Matching and Extraction
- Text Cleaning Pipelines
- Real-World Text Processing Examples

**Time Estimate:** 1 week

**[Complete Guide →](06-regular-expressions-text-processing.md)**

### 07-advanced-data-wrangling
Advanced data manipulation: reshaping, pivoting, and transforming data.

**Topics:**
- Reshaping Data (Wide vs Long Format)
- Pivot Tables and Cross-tabulation
- Melt and Unpivot Operations
- Stack and Unstack
- Advanced Grouping Techniques
- Performance Optimization

**Time Estimate:** 1 week

**[Complete Guide →](07-advanced-data-wrangling.md)**

### 08-working-with-dates-times
Comprehensive guide to handling dates, times, and time-based data.

**Topics:**
- Creating and Parsing Dates
- Date Arithmetic and Differences
- Time Series Indexing
- Extracting Date/Time Components
- Time Zones
- Resampling and Frequency Conversion
- Rolling Windows and Shifting

**Time Estimate:** 1 week

**[Complete Guide →](08-working-with-dates-times.md)**

### 09-streamlit-dashboards
Build interactive web applications and dashboards for data science and machine learning.

**Topics:**
- Streamlit Basics and Components
- Interactive Widgets (Sliders, Dropdowns, Buttons)
- Data Visualization Integration
- Building ML Model Interfaces
- Creating Data Exploration Dashboards
- Deployment and Best Practices

**Time Estimate:** 1 week

**[Complete Guide →](09-streamlit-dashboards.md)**

### 10-flask-web-development
Build web applications and REST APIs with Flask.

**Topics:**
- Flask Basics and Routing
- Templates and Jinja2
- Forms and User Input
- REST API Development
- Database Integration
- Authentication and Sessions
- Deployment
- Flask vs Streamlit Comparison

**Time Estimate:** 1-2 weeks

**[Complete Guide →](10-flask-web-development.md)**

### 11-tableau-visualization
Create professional data visualizations and dashboards with Tableau.

**Topics:**
- Tableau Basics and Interface
- Connecting to Data Sources
- Basic and Advanced Visualizations
- Calculations and Functions
- Dashboards and Stories
- Tableau vs Python Visualization
- Best Practices

**Time Estimate:** 1-2 weeks

**[Complete Guide →](11-tableau-visualization.md)**

## Essential path vs optional depth

Module 01 has **15 lessons**, too much to read linearly before starting ML. Use this split:

### Essential path (~4–6 weeks), required before Module 02

| Order | Lesson | Why |
|-------|--------|-----|
| 1 | [01-numpy.md](01-numpy.md) | Numerical foundation for all ML |
| 2 | [02-pandas.md](02-pandas.md) | Data manipulation |
| 3 | [03-visualization.md](03-visualization.md) | Communicate findings |
| 4 | [04-exploratory-data-analysis.md](04-exploratory-data-analysis.md) | Systematic EDA before modeling |
| 5 | [05-data-sources-and-integration.md](05-data-sources-and-integration.md) | APIs, files, basic SQL hooks |
| 6 | [python-for-data-science-project-tutorial.md](python-for-data-science-project-tutorial.md) | End-to-end practice |

**Parallel (Stage 1.5):** Start [Module 19 SQL](../19-sql-database-fundamentals/README.md) after lesson 2–3 if you target analyst or data scientist roles.

**After first EDA project:** Read [Stakeholder Communication](../resources/stakeholder_communication.md) (30 min) before presenting charts to non-technical audiences.

### Optional depth: revisit when a project needs them

| Lesson | When to study |
|--------|----------------|
| [06-regular-expressions-text-processing.md](06-regular-expressions-text-processing.md) | NLP or messy text columns |
| [07-advanced-data-wrangling.md](07-advanced-data-wrangling.md) | Complex pivots / reshaping |
| [08-working-with-dates-times.md](08-working-with-dates-times.md) | Time series or datetime-heavy data |
| [09-streamlit-dashboards.md](09-streamlit-dashboards.md) | Dashboard portfolio project |
| [10-flask-web-development.md](10-flask-web-development.md) | Before Module 13 deployment |
| [11-tableau-visualization.md](11-tableau-visualization.md) | BI analyst track |
| [python-for-data-science-advanced-topics.md](python-for-data-science-advanced-topics.md) | Polars/Dask at scale |
| [python-for-data-science-quick-reference.md](python-for-data-science-quick-reference.md) | Cheat sheet anytime |

### Exit gate (Stage 1)

Before Module 02, you should pass [Gate B in FOUNDATION_AND_JOB_READINESS.md](../FOUNDATION_AND_JOB_READINESS.md#gate-b--after-module-01-stage-1).

##  Learning Objectives

By the end of this phase, you should be able to:
- Perform numerical operations with NumPy
- Load, clean, and manipulate datasets with Pandas
- Create various types of visualizations (static and interactive)
- Build interactive dashboards with Plotly, Dash, Streamlit, and Tableau
- Create web applications and REST APIs with Flask
- Perform systematic exploratory data analysis (EDA)
- Fetch data from APIs and databases
- Scrape data from websites
- Work with various file formats
- Use regular expressions for text processing
- Reshape and transform data efficiently
- Handle dates and times effectively

##  Projects

1. **Data Analysis Project**: Analyze a real dataset (e.g., sales data, weather data)
2. **Visualization Project**: Create a dashboard with multiple visualizations
3. **Build a machine learning model from scratch**, after NumPy and Pandas, combine skills in an end-to-end numeric pipeline: for example, follow the [Prerequisites Project Tutorial: Neural Network from Scratch](../00-prerequisites/prerequisites-project-tutorial.md) (NumPy only) and/or [First ML Project Tutorial](../02-introduction-to-ml/first-ml-project-tutorial.md) for a full sklearn-style workflow

##  Tips

- Practice with real datasets from Kaggle or UCI Repository
- Focus on understanding data structures (Series, DataFrame)
- Learn to read documentation. It's a crucial skill
- Experiment with different plot types

## Documentation & Learning Resources

### NumPy

**Official Documentation:**
- [NumPy Official Documentation](https://numpy.org/doc/stable/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy API Reference](https://numpy.org/doc/stable/reference/)

**Free Courses & Tutorials:**
- [NumPy Tutorial (W3Schools)](https://www.w3schools.com/python/numpy/)
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Tutorial (DataCamp)](https://www.datacamp.com/tutorial numpy-tutorial-for-beginners): Free tutorial
- [NumPy Basics (Real Python)](https://realpython.com/numpy-tutorial/)

**Video Tutorials:**
- [NumPy Tutorial (Corey Schafer)](https://www.youtube.com/watch?v=QUT1VHiLmmI)
- [NumPy Arrays (Sentdex)](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

### Pandas

**Official Documentation:**
- [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)

**Free Courses & Tutorials:**
- [Pandas Tutorial (W3Schools)](https://www.w3schools.com/python/pandas/)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html): Quick start guide
- [Pandas Tutorial (DataCamp)](https://www.datacamp.com/tutorial/pandas): Free tutorial
- [Pandas Tutorial (Real Python)](https://realpython.com/pandas-python-explore-dataset/)

**Video Tutorials:**
- [Pandas Tutorial (Corey Schafer)](https://www.youtube.com/watch?v=ZyhVh-qRZPA)
- [Pandas for Data Science (Keith Galli)](https://www.youtube.com/watch?v=vmEHCJofslg)

**Practice:**
- [Pandas Exercises (GitHub)](https://github.com/guipsamora/pandas_exercises)
- [Pandas Practice Problems (Kaggle)](https://www.kaggle.com/learn/pandas)

### Matplotlib & Seaborn

**Official Documentation:**
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html): Examples for every plot type
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html): Statistical visualization examples

**Free Courses & Tutorials:**
- [Matplotlib Tutorial (W3Schools)](https://www.w3schools.com/python/matplotlib_intro.asp)
- [Matplotlib Tutorial (Real Python)](https://realpython.com/python-matplotlib-guide/)
- [Seaborn Tutorial (Real Python)](https://realpython.com/python-seaborn-tutorial/)
- [Data Visualization with Python (Coursera)](https://www.coursera.org/learn/python-for-data-visualization): Free audit available

**Video Tutorials:**
- [Matplotlib Tutorial (Corey Schafer)](https://www.youtube.com/watch?v=UO98lJQ3QGI)
- [Seaborn Tutorial (Corey Schafer)](https://www.youtube.com/watch?v=6GUZXDef2U0)
- [Data Visualization (Keith Galli)](https://www.youtube.com/watch?v=0P7QnIQx2IE)

### Additional Resources

- **[Advanced Python for Data Science Topics](python-for-data-science-advanced-topics.md)** - Advanced Pandas techniques, performance optimization, advanced visualization, memory optimization, advanced EDA, data pipeline design, and tool integration
- **[Complete Data Science Project Tutorial](python-for-data-science-project-tutorial.md)** - End-to-end project from data collection (web scraping) to interactive dashboard using all data science tools
- **[Python for Data Science Quick Reference](python-for-data-science-quick-reference.md)** - Quick lookup guide for NumPy, Pandas, visualization, data manipulation patterns, and file I/O

---

**Previous Module:** [00-prerequisites](../00-prerequisites/README.md)  
**Next Module:** [02-introduction-to-ml](../02-introduction-to-ml/README.md)

**Parallel (Stage 1.5):** [19-sql-database-fundamentals](../19-sql-database-fundamentals/README.md) after NumPy/Pandas lessons, for analyst and data scientist tracks.

