# Data Visualization Complete Guide

Comprehensive guide to Matplotlib and Seaborn for creating beautiful and informative visualizations.

## Table of Contents

- [Introduction](#introduction)
- [Matplotlib Basics](#matplotlib-basics)
- [Common Plot Types](#common-plot-types)
- [Customizing Plots](#customizing-plots)
- [Subplots and Multiple Plots](#subplots-and-multiple-plots)
- [Seaborn Statistical Visualizations](#seaborn-statistical-visualizations)
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

## Next Steps

- Practice creating different plot types
- Work with real datasets
- Experiment with customization
- Move to [02-introduction-to-ml](../02-introduction-to-ml/README.md) for ML concepts

**Remember**: Good visualizations tell a story - always think about what message you want to convey!

