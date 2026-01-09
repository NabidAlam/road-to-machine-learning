# Data Visualization Complete Guide

Comprehensive guide to Matplotlib and Seaborn for creating beautiful and informative visualizations.

## Table of Contents

- [Introduction](#introduction)
- [Matplotlib Basics](#matplotlib-basics)
- [Figure and Axes Structure](#figure-and-axes-structure)
- [Common Plot Types](#common-plot-types)
- [Customizing Plots](#customizing-plots)
- [Subplots and Multiple Plots](#subplots-and-multiple-plots)
- [Seaborn Introduction](#seaborn-introduction)
- [Seaborn Relational Plots](#seaborn-relational-plots)
- [Seaborn Categorical Plots](#seaborn-categorical-plots)
- [Seaborn Distribution Plots](#seaborn-distribution-plots)
- [Seaborn Regression & Mixed Plots](#seaborn-regression--mixed-plots)
- [Seaborn Matrix & Styling Plots](#seaborn-matrix--styling-plots)
- [Saving Figures](#saving-figures)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### What is Matplotlib?

Matplotlib is Python's primary plotting library. It provides:
- **Flexibility**: Create any type of plot
- **Control**: Fine-grained control over every element
- **Publication quality**: High-quality figures for papers

### What is Seaborn?

Seaborn is built on Matplotlib and provides:
- **Statistical plots**: Built-in statistical visualizations
- **Beautiful defaults**: Attractive styling out of the box
- **Easy to use**: Less code for common plots

### Installation

```python
pip install matplotlib seaborn
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')  # or 'default', 'ggplot', etc.
sns.set_style("whitegrid")
```

---

## Matplotlib Basics

### Simple Plot

```python
# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Sine Wave')
plt.grid(True)
plt.show()
```

### Multiple Lines

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine and Cosine')
plt.legend()
plt.grid(True)
plt.show()
```

### Figure and Axes

```python
# Explicit figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Using Axes Object')
ax.grid(True)
plt.show()
```

---

## Figure and Axes Structure

Understanding the structure of a Matplotlib figure is crucial for creating professional visualizations.

### Components

**Figure (Canvas):**
- The entire window or page that everything is drawn on
- Can contain multiple Axes (subplots)
- Controls overall size, DPI, and background

**Axes (Plot Area):**
- The actual plot area where data is visualized
- Contains the plot, labels, legend, etc.
- A Figure can contain multiple Axes

**Axis (x/y scales):**
- The number-line-like objects that provide ticks and labels
- X-axis and Y-axis define the coordinate system

```python
# Create figure and axes explicitly
fig = plt.figure(figsize=(10, 6))  # Figure (canvas)
ax = fig.add_subplot(111)          # Axes (plot area)

# Or use the convenient method
fig, ax = plt.subplots(figsize=(10, 6))

# Access axes
ax.plot(x, y)
ax.set_xlabel('X Axis')            # X-axis label
ax.set_ylabel('Y Axis')            # Y-axis label
ax.set_title('Title')              # Title on axes

# Multiple axes in one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(x, y1)
ax2.plot(x, y2)
```

### Understanding the Hierarchy

```python
# Figure level
fig = plt.figure(figsize=(10, 6), dpi=100)  # DPI = dots per inch
fig.suptitle('Figure Title', fontsize=16)    # Title for entire figure

# Axes level
ax = fig.add_subplot(111)
ax.set_title('Axes Title')                   # Title for this plot
ax.set_xlabel('X Label')                     # X-axis label
ax.set_ylabel('Y Label')                     # Y-axis label

# Axis level (access through axes)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Control x-axis ticks
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))  # Format y-axis
```

---

## Common Plot Types

### Line Plot

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue', linestyle='-', marker='o', markersize=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()
```

### Scatter Plot

```python
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
```

### Bar Plot

```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.show()

# Horizontal bar
plt.figure(figsize=(10, 6))
plt.barh(categories, values, color='lightcoral')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Plot')
plt.show()
```

### Histogram

```python
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Multiple histograms
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(120, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, alpha=0.5, label='Group 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Group 2', color='red')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Multiple Histograms')
plt.legend()
plt.show()
```

### Box Plot

```python
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'])
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```

### Pie Chart

```python
sizes = [30, 25, 25, 20]
labels = ['A', 'B', 'C', 'D']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart')
plt.show()
```

---

## Customizing Plots

### Colors and Styles

```python
x = np.linspace(0, 10, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), color='red', linewidth=2, linestyle='--', label='sin(x)')
plt.plot(x, np.cos(x), color='#0066CC', linewidth=2, linestyle='-', marker='o', markersize=3, label='cos(x)')
plt.xlabel('X', fontsize=12, fontweight='bold')
plt.ylabel('Y', fontsize=12, fontweight='bold')
plt.title('Customized Plot', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()
```

### Axis Customization

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
ax.set_xlabel('X axis', fontsize=12)
ax.set_ylabel('Y axis', fontsize=12)
ax.set_title('Customized Axes', fontsize=14)
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(-1, 2, 0.5))
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

### Text and Annotations

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot with Annotations')

# Add text
plt.text(5, 0.5, 'Peak', fontsize=12, ha='center')

# Add annotation with arrow
plt.annotate('Maximum', xy=(np.pi/2, 1), xytext=(4, 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12)

plt.show()
```

---

## Subplots and Multiple Plots

### Subplots

```python
# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
axes[0, 0].plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True)

# Plot 2
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100))
axes[0, 1].set_title('Scatter Plot')

# Plot 3
axes[1, 0].bar(['A', 'B', 'C', 'D'], [10, 20, 30, 40])
axes[1, 0].set_title('Bar Plot')

# Plot 4
axes[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### GridSpec (Advanced Layout)

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
ax3 = fig.add_subplot(gs[1, 1])  # Bottom right

ax1.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
ax2.scatter(np.random.randn(50), np.random.randn(50))
ax3.bar(['A', 'B', 'C'], [10, 20, 30])

plt.tight_layout()
plt.show()
```

---

## Seaborn Introduction

### What is Seaborn?

Seaborn is a high-level visualization library built on top of Matplotlib, designed for quick, attractive, and informative statistical plots with minimal code.

**Key Features:**
- **Statistical plots**: Built-in statistical visualizations
- **Beautiful defaults**: Attractive styling out of the box
- **Easy to use**: Less code for common plots
- **Integration**: Works seamlessly with Pandas DataFrames

### Installation and Setup

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style globally
sns.set_style("whitegrid")  # Options: darkgrid, whitegrid, dark, white, ticks
sns.set_palette("husl")     # Set color palette
```

---

## Seaborn Relational Plots

Relational plots visualize relationships between two variables, helping identify trends, correlations, and distributions.

### scatterplot()

Displays individual data points to show correlation:

```python
# Load sample data
tips = sns.load_dataset('tips')

# Basic scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Total Bill vs Tip')
plt.show()

# With hue (color by category)
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.show()

# With size
sns.scatterplot(data=tips, x='total_bill', y='tip', 
                hue='sex', size='size', sizes=(50, 200))
plt.show()

# With style
sns.scatterplot(data=tips, x='total_bill', y='tip', 
                hue='sex', style='time')
plt.show()
```

**Use Cases:**
- Visualize relationships between numeric variables (price vs. sales)
- Discover patterns in data
- Identify correlations

### lineplot()

Shows continuous trends or summary lines over time or categories:

```python
# Basic line plot
sns.lineplot(data=tips, x='total_bill', y='tip')
plt.show()

# With hue
sns.lineplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.show()

# With markers
sns.lineplot(data=tips, x='total_bill', y='tip', 
             hue='sex', marker='o', markersize=8)
plt.show()

# Time series example
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(100))
})
sns.lineplot(data=ts_data, x='date', y='value')
plt.show()
```

**Use Cases:**
- Show trends over time
- Compare trends across categories
- Visualize continuous relationships

### Styling & Palettes

Customize visual appeal with themes and color palettes:

```python
# Set style
sns.set_style("darkgrid")    # Dark background with grid
sns.set_style("whitegrid")   # White background with grid
sns.set_style("dark")         # Dark background, no grid
sns.set_style("white")        # White background, no grid
sns.set_style("ticks")        # Minimal style with ticks

# Set color palette
sns.set_palette("pastel")     # Soft, muted colors
sns.set_palette("deep")       # Deep, rich colors
sns.set_palette("muted")       # Muted, balanced colors
sns.set_palette("bright")      # Bright, vibrant colors
sns.set_palette("dark")        # Dark colors
sns.set_palette("colorblind")  # Colorblind-friendly

# Custom palette
custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
sns.set_palette(custom_palette)

# Reset to default
sns.reset_orig()  # Reset to matplotlib defaults
```

---

## Seaborn Categorical Plots

Categorical plots are used when one variable is categorical (like gender, day, or class). They help compare counts, averages, and spread across different groups.

### barplot()

Displays the average of a numeric variable for each category:

```python
# Basic bar plot
sns.barplot(data=tips, x='day', y='total_bill')
plt.title('Average Bill by Day')
plt.show()

# With hue (grouping)
sns.barplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

# Horizontal bar plot
sns.barplot(data=tips, x='total_bill', y='day', orient='h')
plt.show()

# Custom estimator (e.g., median instead of mean)
sns.barplot(data=tips, x='day', y='total_bill', estimator=np.median)
plt.show()
```

### countplot()

Shows the frequency of observations per category:

```python
# Basic count plot
sns.countplot(data=tips, x='day')
plt.title('Frequency by Day')
plt.show()

# With hue
sns.countplot(data=tips, x='day', hue='sex')
plt.show()

# Horizontal
sns.countplot(data=tips, y='day')
plt.show()

# Order categories
sns.countplot(data=tips, x='day', order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.show()
```

### boxplot()

Visualizes median, quartiles, and outliers:

```python
# Basic box plot
sns.boxplot(data=tips, x='day', y='total_bill')
plt.title('Bill Distribution by Day')
plt.show()

# With hue
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

# Horizontal
sns.boxplot(data=tips, x='total_bill', y='day', orient='h')
plt.show()

# Customize
sns.boxplot(data=tips, x='day', y='total_bill', 
            palette='Set2', linewidth=2)
plt.show()
```

### violinplot()

Combines boxplot with distribution shape for deeper insight:

```python
# Basic violin plot
sns.violinplot(data=tips, x='day', y='total_bill')
plt.title('Distribution Shape by Day')
plt.show()

# With hue
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

# Split violins
sns.violinplot(data=tips, x='day', y='total_bill', 
               hue='sex', split=True)
plt.show()

# Inner plot type
sns.violinplot(data=tips, x='day', y='total_bill', 
               inner='box')  # or 'quart', 'point', 'stick'
plt.show()
```

### stripplot()

Displays all data points, even if they overlap:

```python
# Basic strip plot
sns.stripplot(data=tips, x='day', y='total_bill')
plt.title('All Data Points by Day')
plt.show()

# With jitter (reduce overlap)
sns.stripplot(data=tips, x='day', y='total_bill', jitter=True)
plt.show()

# With hue
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

# Customize
sns.stripplot(data=tips, x='day', y='total_bill', 
              size=4, alpha=0.5, palette='Set2')
plt.show()
```

### swarmplot()

Similar to stripplot but prevents overlap of points:

```python
# Basic swarm plot
sns.swarmplot(data=tips, x='day', y='total_bill')
plt.title('Non-Overlapping Points by Day')
plt.show()

# With hue
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

# Combined with violin plot
sns.violinplot(data=tips, x='day', y='total_bill', inner=None)
sns.swarmplot(data=tips, x='day', y='total_bill', 
              color='white', edgecolor='gray')
plt.show()
```

**Practical Use Cases:**
- Compare customer spending by weekday
- Analyze score distributions by gender
- Visualize category frequencies in survey or sales data

---

## Seaborn Distribution Plots

Distribution plots visualize how data values are spread, concentrated, or distributed across a range.

### histplot()

Modern histogram with options for bins and KDE overlay:

```python
# Basic histogram
sns.histplot(data=tips, x='total_bill', bins=30)
plt.title('Distribution of Total Bill')
plt.show()

# With KDE overlay
sns.histplot(data=tips, x='total_bill', bins=30, kde=True)
plt.show()

# With hue (compare multiple groups)
sns.histplot(data=tips, x='total_bill', hue='sex', bins=30, kde=True)
plt.show()

# Stacked histogram
sns.histplot(data=tips, x='total_bill', hue='sex', 
             bins=30, multiple='stack')
plt.show()

# Step histogram
sns.histplot(data=tips, x='total_bill', bins=30, 
             element='step', fill=False)
plt.show()
```

### kdeplot()

Displays a smooth density curve representing data distribution:

```python
# Basic KDE plot
sns.kdeplot(data=tips, x='total_bill')
plt.title('Density Curve of Total Bill')
plt.show()

# With shade
sns.kdeplot(data=tips, x='total_bill', shade=True)
plt.show()

# With hue
sns.kdeplot(data=tips, x='total_bill', hue='sex', shade=True)
plt.show()

# 2D KDE plot
sns.kdeplot(data=tips, x='total_bill', y='tip', shade=True)
plt.show()

# Cumulative distribution
sns.kdeplot(data=tips, x='total_bill', cumulative=True)
plt.show()
```

### rugplot()

Adds small tick marks for individual observations:

```python
# Basic rug plot
sns.rugplot(data=tips, x='total_bill')
plt.show()

# Combined with histogram and KDE
sns.histplot(data=tips, x='total_bill', bins=30, kde=True)
sns.rugplot(data=tips, x='total_bill', height=0.05, alpha=0.5)
plt.show()
```

### Note on distplot()

**`distplot()` is deprecated** - replaced by `histplot()` and `kdeplot()`:

```python
# OLD (deprecated):
# sns.distplot(tips['total_bill'])

# NEW (use instead):
sns.histplot(data=tips, x='total_bill', kde=True)  # Histogram with KDE
# or
sns.kdeplot(data=tips, x='total_bill')  # Just KDE
```

**Advanced Use with hue:**

```python
# Compare multiple groups
sns.histplot(data=tips, x='total_bill', hue='sex', 
             bins=30, kde=True, alpha=0.6)
plt.title('Distribution Comparison: Male vs Female Spending')
plt.show()
```

**Practical Use Cases:**
- Analyze distribution of prices, customer tips, or sales amounts
- Compare value distributions across categories for insights
- Identify data patterns and outliers

---

## Seaborn Regression & Mixed Plots

Regression and mixed plots visualize relationships between variables along with trend lines or combined plot types.

### regplot()

Displays a scatterplot with a fitted regression line:

```python
# Basic regression plot
sns.regplot(data=tips, x='total_bill', y='tip')
plt.title('Total Bill vs Tip with Regression Line')
plt.show()

# With confidence interval
sns.regplot(data=tips, x='total_bill', y='tip', ci=95)
plt.show()

# Without confidence interval
sns.regplot(data=tips, x='total_bill', y='tip', ci=None)
plt.show()

# Customize markers
sns.regplot(data=tips, x='total_bill', y='tip', 
            marker='+', scatter_kws={'s': 50, 'alpha': 0.5})
plt.show()

# Polynomial regression
sns.regplot(data=tips, x='total_bill', y='tip', order=2)
plt.show()
```

### lmplot()

Similar to regplot but supports grouping (hue) and faceting:

```python
# Basic lmplot
sns.lmplot(data=tips, x='total_bill', y='tip')
plt.show()

# With hue (grouping)
sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.show()

# With faceting (separate plots)
sns.lmplot(data=tips, x='total_bill', y='tip', 
           col='day', col_wrap=2)
plt.show()

# Multiple facets
sns.lmplot(data=tips, x='total_bill', y='tip', 
           row='sex', col='day')
plt.show()

# Customize
sns.lmplot(data=tips, x='total_bill', y='tip', 
           hue='sex', markers=['o', 's'], palette='Set1')
plt.show()
```

### jointplot()

Combines scatter and histogram/KDE for joint and marginal distributions:

```python
# Basic joint plot (scatter + histograms)
sns.jointplot(data=tips, x='total_bill', y='tip')
plt.show()

# With KDE
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde')
plt.show()

# With regression
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.show()

# With hex bins
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
plt.show()

# With scatter and KDE
sns.jointplot(data=tips, x='total_bill', y='tip', 
              kind='scatter', marginal_kws=dict(bins=30, kde=True))
plt.show()
```

### pairplot()

Shows pairwise relationships among multiple variables in a dataset:

```python
# Basic pair plot
iris = sns.load_dataset('iris')
sns.pairplot(iris)
plt.show()

# With hue (grouping)
sns.pairplot(iris, hue='species')
plt.show()

# Customize diagonal
sns.pairplot(iris, hue='species', diag_kind='kde')
plt.show()

# Select specific variables
sns.pairplot(iris, vars=['sepal_length', 'sepal_width', 'petal_length'], 
             hue='species')
plt.show()

# Customize markers
sns.pairplot(iris, hue='species', markers=['o', 's', 'D'])
plt.show()
```

**Practical Use Cases:**
- Analyze correlations (e.g., total bill vs tip)
- Visualize gender-based trends
- Explore relationships among multiple numeric features in datasets like Iris

---

## Seaborn Matrix & Styling Plots

Matrix or grid plots are great for exploring correlations, patterns, and relationships within datasets.

### heatmap()

Displays correlations or pivot tables as colored grids:

```python
# Correlation heatmap
correlation = tips.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pivot table heatmap
pivot = tips.pivot_table(values='total_bill', index='day', columns='time')
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Total Bill Heatmap')
plt.show()

# Customize
sns.heatmap(correlation, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.show()

# Without annotations
sns.heatmap(correlation, annot=False, cmap='viridis')
plt.show()
```

### clustermap()

Adds clustering to heatmaps for grouping similar variables or patterns:

```python
# Basic clustermap
sns.clustermap(correlation, annot=True, cmap='coolwarm', center=0)
plt.show()

# With different metric
sns.clustermap(correlation, metric='euclidean', method='ward')
plt.show()

# Customize
sns.clustermap(correlation, annot=True, fmt='.2f',
               cmap='coolwarm', figsize=(10, 8),
               row_cluster=True, col_cluster=True)
plt.show()
```

### Styling Options

Control global styling for all plots:

```python
# Set style
sns.set_style("white")        # White background
sns.set_style("dark")         # Dark background
sns.set_style("whitegrid")    # White with grid (default)
sns.set_style("darkgrid")     # Dark with grid
sns.set_style("ticks")        # Minimal with ticks

# Set palette
sns.set_palette("Set2")       # Qualitative palette
sns.set_palette("pastel")     # Soft colors
sns.set_palette("deep")       # Deep colors
sns.set_palette("muted")      # Muted colors
sns.set_palette("bright")     # Bright colors
sns.set_palette("dark")       # Dark colors
sns.set_palette("colorblind") # Colorblind-friendly

# Set context (affects scale)
sns.set_context("paper")      # Smallest (for papers)
sns.set_context("notebook")   # Default
sns.set_context("talk")       # Larger (for presentations)
sns.set_context("poster")     # Largest (for posters)

# Combine settings
sns.set_style("whitegrid")
sns.set_palette("Set2")
sns.set_context("notebook")

# Reset to defaults
sns.reset_orig()
```

**Use Cases:**
- Analyze correlations between variables
- Highlight variable groupings
- Create visually consistent plots with global themes

---

## Seaborn Statistical Visualizations

### Distribution Plots

```python
# Load sample data
tips = sns.load_dataset('tips')

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, bins=30)
plt.title('Distribution of Total Bill')
plt.show()

# KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=tips, x='total_bill', shade=True)
plt.title('KDE Plot')
plt.show()
```

### Categorical Plots

```python
# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Average Bill by Day and Gender')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Bill Distribution by Day')
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill')
plt.title('Violin Plot')
plt.show()
```

### Relationship Plots

```python
# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', size='size')
plt.title('Tip vs Total Bill')
plt.show()

# Regression plot
plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip')
plt.title('Regression Plot')
plt.show()

# Pair plot (multiple relationships)
sns.pairplot(tips, hue='sex', diag_kind='kde')
plt.show()
```

### Heatmap

```python
# Correlation heatmap
correlation = tips.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pivot table heatmap
pivot = tips.pivot_table(values='total_bill', index='day', columns='time')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Heatmap of Total Bill')
plt.show()
```

### Facet Grid

```python
# Create facet grid
g = sns.FacetGrid(tips, col='day', row='sex', height=4)
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()
plt.show()
```

---

## Saving Figures

### Saving Plots

```python
# Create plot
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
plt.title('Sine Wave')

# Save with different formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # PNG
plt.savefig('plot.pdf', bbox_inches='tight')           # PDF
plt.savefig('plot.svg', bbox_inches='tight')          # SVG
plt.savefig('plot.jpg', dpi=300, quality=95)          # JPEG

# With transparent background
plt.savefig('plot.png', transparent=True, dpi=300)

plt.show()
```

### High-Quality Figures

```python
# For publications
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(x, y, linewidth=2)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_title('Publication Quality Figure', fontsize=16)
plt.tight_layout()
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Practice Exercises

### Exercise 1: Basic Line Plot

**Task:** Create a line plot of y = x² from -5 to 5 with proper labels and grid.

**Solution:**
```python
x = np.linspace(-5, 5, 100)
y = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y = X²', fontsize=12)
plt.title('Quadratic Function', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

### Exercise 2: Multiple Subplots

**Task:** Create a 2x2 subplot with different plot types.

**Solution:**
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')

# Scatter plot
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100))
axes[0, 1].set_title('Scatter Plot')

# Bar plot
axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 30])
axes[1, 0].set_title('Bar Plot')

# Histogram
axes[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### Exercise 3: Data Visualization

**Task:** Load a dataset and create visualizations showing distributions and relationships.

**Solution:**
```python
# Using tips dataset
tips = sns.load_dataset('tips')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution
sns.histplot(data=tips, x='total_bill', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Total Bill')

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0, 1])
axes[0, 1].set_title('Bill by Day')

# Scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', ax=axes[1, 0])
axes[1, 0].set_title('Tip vs Bill')

# Bar plot
sns.barplot(data=tips, x='day', y='total_bill', ax=axes[1, 1])
axes[1, 1].set_title('Average Bill by Day')

plt.tight_layout()
plt.show()
```

### Exercise 4: Correlation Heatmap

**Task:** Create a correlation heatmap for numeric columns in a dataset.

**Solution:**
```python
# Calculate correlation
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100),
    'D': np.random.randn(100)
})

correlation = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Correlation Heatmap')
plt.show()
```

---

## Key Takeaways

1. **Matplotlib**: Full control, use for custom plots
2. **Seaborn**: Beautiful defaults, use for statistical plots
3. **Subplots**: Organize multiple plots
4. **Customization**: Always label axes and add titles
5. **Save high-quality**: Use high DPI for publications

---

## Common Patterns

### Pattern 1: EDA Visualization

```python
def plot_eda(df):
    """Create comprehensive EDA visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution
    axes[0, 0].hist(df['column'], bins=30)
    axes[0, 0].set_title('Distribution')
    
    # Box plot
    axes[0, 1].boxplot(df['column'])
    axes[0, 1].set_title('Box Plot')
    
    # Scatter
    axes[1, 0].scatter(df['x'], df['y'])
    axes[1, 0].set_title('Scatter Plot')
    
    # Correlation
    sns.heatmap(df.corr(), annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation')
    
    plt.tight_layout()
    plt.show()
```

---

## Plotly and Dash for Interactive Visualizations

Plotly is a powerful library for creating interactive, publication-quality visualizations. Dash is a framework built on Plotly for building analytical web applications.

### Why Plotly?

**Advantages:**
- **Interactive**: Hover, zoom, pan, and filter
- **Publication Quality**: Professional-looking plots
- **Web-Ready**: Works in Jupyter notebooks and web browsers
- **Multiple Backends**: Python, R, JavaScript
- **Rich Features**: 3D plots, animations, subplots

### Installation

```bash
pip install plotly dash
```

### Plotly Express (Simple API)

Plotly Express provides a simple, high-level interface for creating plots.

#### Basic Plots

```python
import plotly.express as px
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.rand(100) * 100
})

# Scatter plot
fig = px.scatter(df, x='x', y='y', color='category', size='size',
                 title='Interactive Scatter Plot',
                 labels={'x': 'X Axis', 'y': 'Y Axis'})
fig.show()

# Line plot
df_time = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'value': np.cumsum(np.random.randn(100))
})
fig = px.line(df_time, x='date', y='value', title='Time Series')
fig.show()

# Bar chart
df_bar = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [10, 20, 15, 25]
})
fig = px.bar(df_bar, x='category', y='value', title='Bar Chart')
fig.show()

# Histogram
fig = px.histogram(df, x='x', nbins=30, title='Histogram')
fig.show()

# Box plot
fig = px.box(df, x='category', y='y', title='Box Plot')
fig.show()
```

#### Advanced Plotly Express

```python
# 3D Scatter
fig = px.scatter_3d(df, x='x', y='y', z='size', color='category')
fig.show()

# Faceted plots
fig = px.scatter(df, x='x', y='y', facet_col='category', 
                 title='Faceted Scatter Plot')
fig.show()

# Heatmap
correlation_matrix = df[['x', 'y', 'size']].corr()
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                title='Correlation Heatmap')
fig.show()
```

### Plotly Graph Objects (Advanced Control)

For more control, use Plotly Graph Objects:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='lines+markers',
    name='Line 1'
))

# Update layout
fig.update_layout(
    title='Custom Plot',
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    hovermode='x unified'
)

fig.show()
```

### Dash for Web Applications

Dash allows you to build interactive web applications with Plotly.

#### Basic Dash App

```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Create Dash app
app = dash.Dash(__name__)

# Create figure
fig = px.scatter(df, x='x', y='y', color='category')

# Define app layout
app.layout = html.Div([
    html.H1("My Dash App"),
    dcc.Graph(figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
```

### Key Takeaways for Plotly and Dash

1. **Plotly Express**: Simple API for quick interactive plots
2. **Plotly Graph Objects**: Advanced control and customization
3. **Dash**: Build interactive web applications
4. **Interactivity**: Hover, zoom, pan, filter built-in
5. **Deployment**: Deploy Dash apps to cloud platforms

---

## Next Steps

- Practice creating different plot types
- Work with real datasets
- Experiment with customization
- Try Plotly for interactive visualizations
- Build a Dash dashboard for your project
- Move to [02-introduction-to-ml](../02-introduction-to-ml/README.md) for ML concepts

**Remember**: Good visualizations tell a story - always think about what message you want to convey! Plotly and Dash take your visualizations to the next level with interactivity and web deployment.

