import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

base = Path(__file__).resolve().parent.parent

data = pd.read_csv(base / "data" / "GCC_DEMO.csv")

# -------------------------------------------------------
# Variables
# -------------------------------------------------------

independent_variables = [
    "DES",
    "Power",
    "Time"
]

responses = [
    "TPC",
    "TFC",
    "DPPH",
    "ABTS",
    "MCA"
]

# -------------------------------------------------------
# Regression Loop
# -------------------------------------------------------

results = []

for x_variable in independent_variables:

    for response in responses:

        X = data[[x_variable]]

        y = data[response]

        model = LinearRegression()

        model.fit(X, y)

        prediction = model.predict(X)

        r2 = r2_score(y, prediction)

        slope = model.coef_[0]

        intercept = model.intercept_

        results.append({

            "Independent Variable": x_variable,

            "Response": response,

            "Slope": slope,

            "Intercept": intercept,

            "R2": r2

        })

        plt.figure(figsize=(6,5))

        plt.scatter(X, y)

        plt.plot(X, prediction)

        plt.xlabel(x_variable)

        plt.ylabel(response)

        plt.title(f"{response} vs {x_variable}")

        filename = f"{response}_vs_{x_variable}_Regression.png"

        plt.savefig(

            base/"results"/filename,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

# -------------------------------------------------------
# Save Results
# -------------------------------------------------------

results = pd.DataFrame(results)

print(results)

results.to_csv(

    base/"results"/"Regression_Results.csv",

    index=False

)

print("\nRegression analysis completed.")
