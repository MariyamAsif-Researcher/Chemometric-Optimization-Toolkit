import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC", "TFC", "DPPH", "ABTS", "MCA"]

for variable in responses:

    plt.figure(figsize=(6,4))

    plt.boxplot(data[variable])

    plt.title(variable + " Boxplot")

    plt.ylabel(variable)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig("figures/" + variable + "_boxplot.png", dpi=300)

    plt.close()

print("All boxplots created successfully!")