import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC", "TFC", "DPPH", "ABTS", "MCA"]

for variable in responses:

    plt.figure(figsize=(7,5))

    plt.hist(
        data[variable],
        bins=6,
        edgecolor="black"
    )

    plt.title(f"{variable} Distribution", fontsize=14)
    plt.xlabel(variable, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"figures/{variable}_histogram.png", dpi=300)

    plt.close()

print("All histograms saved successfully.")