from pathlib import Path
import pandas as pd

# Locate the dataset
current_folder = Path(__file__).parent
csv_file = current_folder.parent / "data" / "GCC_DEMO.csv"

# Load dataset
data = pd.read_csv(csv_file)

# Generate descriptive statistics
statistics = data.describe()

# Display statistics
print(statistics)

# Save statistics
output = current_folder.parent / "results" / "statistics.csv"
statistics.to_csv(output)

print("\nStatistics saved successfully!")