import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

base = Path(__file__).resolve().parent.parent

data = pd.read_csv(base / "data" / "GCC_DEMO.csv")

# ----------------------------------------------------
# Variables
# ----------------------------------------------------

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

results = []

# ----------------------------------------------------
# Polynomial Regression
# ----------------------------------------------------

for x_variable in independent_variables:

    for response in responses:

        X = data[[x_variable]]

        y = data[response]

        poly = PolynomialFeatures(degree=2)

        X_poly = poly.fit_transform(X)

        model = LinearRegression()

        model.fit(X_poly, y)

        prediction = model.predict(X_poly)

        r2 = r2_score(y, prediction)

        results.append({

            "Independent Variable":x_variable,

            "Response":response,

            "R2":r2

        })

        # Sorting for smooth curve

        sorted_index = X[x_variable].argsort()

        X_sorted = X.iloc[sorted_index]

        prediction_sorted = prediction[sorted_index]

        plt.figure(figsize=(6,5))

        plt.scatter(X,y)

        plt.plot(

            X_sorted,

            prediction_sorted,

            linewidth=2

        )

        plt.xlabel(x_variable)

        plt.ylabel(response)

        plt.title(

            f"Polynomial Regression\n{response} vs {x_variable}"

        )

        filename = (

            response

            + "_"

            + x_variable

            + "_Polynomial.png"

        )

        plt.savefig(

            base/"results"/filename,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

results = pd.DataFrame(results)

results.to_csv(

    base/"results"/"Polynomial_Regression_Results.csv",

    index=False

)

print(results)