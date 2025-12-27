"""
Python for Data Science - Basic Examples
Quick reference for NumPy, Pandas, and Matplotlib basics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("NUMPY BASICS")
print("=" * 60)

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n1D Array: {arr1}")
print(f"\n2D Array:\n{arr2}")

# Array operations
print(f"\nArray sum: {arr1.sum()}")
print(f"Array mean: {arr1.mean()}")
print(f"Array max: {arr1.max()}")

# Array operations
arr3 = arr1 * 2
print(f"\nArray multiplied by 2: {arr3}")

print("\n" + "=" * 60)
print("PANDAS BASICS")
print("=" * 60)

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris']
}
df = pd.DataFrame(data)
print(f"\nDataFrame:\n{df}")

# Basic operations
print(f"\nDataFrame Info:")
print(df.info())
print(f"\nDataFrame Description:")
print(df.describe())

# Selecting data
print(f"\nSelecting 'Name' column:\n{df['Name']}")
print(f"\nFirst 2 rows:\n{df.head(2)}")

print("\n" + "=" * 60)
print("MATPLOTLIB BASICS")
print("=" * 60)

# Simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Plot Example')
plt.legend()
plt.grid(True)
plt.savefig('simple_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'simple_plot.png'")

print("\n" + "=" * 60)
print("BASIC EXAMPLES COMPLETE!")
print("=" * 60)

