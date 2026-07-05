import pandas as pd
import matplotlib.pyplot as plt

# Read ANN results
ann = pd.read_csv("results/ANN_Performance.csv")

# Read RSM results
rsm = pd.read_csv("results/RSM_Performance.csv")

# Merge both tables
comparison = pd.merge(
    ann,
    rsm,
    on="Response",
    suffixes=("_ANN", "_RSM")
)

# Save merged table
comparison.to_csv(
    "results/Model_Comparison.csv",
    index=False
)

print(comparison)

# --------------------------
# Plot R² comparison
# --------------------------

responses = comparison["Response"]

plt.figure(figsize=(8,5))

plt.plot(
    responses,
    comparison["R2_ANN"],
    marker="o",
    linewidth=2,
    label="ANN"
)

plt.plot(
    responses,
    comparison["R2_RSM"],
    marker="s",
    linewidth=2,
    label="RSM"
)

plt.title("ANN vs RSM (R² Comparison)")

plt.xlabel("Responses")

plt.ylabel("R²")

plt.ylim(0.90,1.01)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("results/R2_Comparison.png",dpi=300)

plt.show()