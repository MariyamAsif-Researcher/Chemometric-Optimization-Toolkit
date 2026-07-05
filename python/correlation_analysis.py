# ==========================================================
# Correlation Analysis
# Chemometric Optimization Toolkit
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# Load Dataset
# --------------------------------------------

data = pd.read_csv("data/GCC_DEMO.csv")

# --------------------------------------------
# Calculate Correlation Matrix
# --------------------------------------------

correlation = data.corr(numeric_only=True)

print("\nCorrelation Matrix\n")
print(correlation)

# --------------------------------------------
# Save correlation values
# --------------------------------------------

correlation.to_csv("results/correlation_matrix.csv")

print("\nCorrelation matrix saved successfully.")

# --------------------------------------------
# Draw Heatmap
# --------------------------------------------

fig, ax = plt.subplots(figsize=(10,8))

image = ax.imshow(correlation,
                  cmap="coolwarm",
                  vmin=-1,
                  vmax=1)

# Labels

ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))

ax.set_xticklabels(correlation.columns,
                   rotation=45,
                   ha="right")

ax.set_yticklabels(correlation.columns)

# Write values inside cells

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        ax.text(j,
                i,
                f"{correlation.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8)

# Color bar

plt.colorbar(image)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("results/correlation_heatmap.png",
            dpi=300)

plt.show()

print("\nHeatmap saved successfully.")

       
