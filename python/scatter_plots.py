import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC", "TFC", "DPPH", "ABTS", "MCA"]

for response in responses:

    plt.figure(figsize=(6,4))

    plt.scatter(data["Power"],
                data[response])

    plt.xlabel("Microwave Power (W)")
    plt.ylabel(response)

    plt.title(f"Microwave Power vs {response}")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/{response}_scatter.png",
        dpi=300
    )

    plt.close()

print("All scatter plots created successfully.")