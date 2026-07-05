import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/GCC_DEMO.csv")

independent = ["DES","Power","Time"]

responses = ["TPC","TFC","DPPH","ABTS","MCA"]

for x in independent:

    for y in responses:

        plt.figure(figsize=(6,4))

        plt.scatter(data[x], data[y])

        plt.title(f"{x} vs {y}")

        plt.xlabel(x)

        plt.ylabel(y)

        plt.grid(True)

        plt.savefig(f"results/{x}_{y}.png")

        plt.close()

print("All scatter plots saved.")