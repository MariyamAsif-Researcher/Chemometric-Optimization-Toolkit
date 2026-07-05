import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/GCC_DEMO.csv")

# Plot histogram
plt.figure(figsize=(8,5))

plt.hist(data["TPC"], bins=6)

plt.title("Distribution of Total Phenolic Content")

plt.xlabel("TPC (mg GAE/g DW)")

plt.ylabel("Frequency")

plt.savefig("results/TPC_histogram.png", dpi=300)

plt.show()