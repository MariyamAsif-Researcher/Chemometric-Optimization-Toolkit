from pathlib import Path
import pandas as pd

# Get the folder where this Python file is located
current_folder = Path(__file__).parent

# Build the path to the data folder
csv_file = current_folder.parent / "data" / "GCC_DEMO.csv"

# Read the dataset
data = pd.read_csv(csv_file)

# Show first five rows
print(data.head())