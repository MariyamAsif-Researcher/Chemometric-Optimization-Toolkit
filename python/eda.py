import pandas as pd

# Load dataset
data = pd.read_csv("data/GCC_DEMO.csv")

# Show first rows
print(data.head())

print("\n")

# Dataset dimensions
print("Rows and Columns")
print(data.shape)

print("\n")

# Column names
print("Variables")
print(data.columns)

print("\n")

# Data types
print("Data Types")
print(data.dtypes)