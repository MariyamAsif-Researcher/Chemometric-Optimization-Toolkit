import pandas as pd

# Load the dataset
data = pd.read_csv("data/GCC_DEMO.csv")

print("\n===== Missing Values =====\n")
print(data.isnull().sum())

print("\n===== Duplicate Rows =====\n")
print(data.duplicated().sum())