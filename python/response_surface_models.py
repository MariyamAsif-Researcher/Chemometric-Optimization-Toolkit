import pandas as pd
import statsmodels.formula.api as smf
import os

# -----------------------------
# Create Results Folder
# -----------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/GCC_DEMO.csv")

responses = [
    "TPC",
    "TFC",
    "DPPH",
    "ABTS",
    "MCA"
]

# -----------------------------
# Fit Response Surface Models
# -----------------------------
for response in responses:

    print("\n")
    print("="*60)
    print("Response:", response)
    print("="*60)

    formula = (
        f"{response} ~ "
        "DES + Power + Time + "
        "I(DES**2) + I(Power**2) + I(Time**2) + "
        "DES:Power + DES:Time + Power:Time"
    )

    model = smf.ols(formula, data=data).fit()

    print(model.summary())

    filename = f"results/{response}_RSM_summary.txt"

    with open(filename, "w") as f:
        f.write(model.summary().as_text())

print("\n")
print("All response surface summaries have been saved successfully.")