import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.formula.api import ols

# --------------------------

data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC","TFC","DPPH","ABTS","MCA"]

for response in responses:

    formula = f"{response} ~ DES + Power + Time + I(DES**2) + I(Power**2) + I(Time**2) + DES:Power + DES:Time + Power:Time"

    model = ols(formula,data=data).fit()

    DES = np.linspace(60,90,100)

    Power = np.linspace(220,440,100)

    DES_grid, Power_grid = np.meshgrid(DES,Power)

    Time = np.full_like(DES_grid,30)

    prediction = pd.DataFrame({

        "DES":DES_grid.ravel(),

        "Power":Power_grid.ravel(),

        "Time":Time.ravel()

    })

    Z = model.predict(prediction)

    Z = Z.values.reshape(DES_grid.shape)

    plt.figure(figsize=(8,6))

    contour = plt.contourf(

        DES_grid,

        Power_grid,

        Z,

        levels=20,

        cmap="viridis"

    )

    plt.xlabel("DES Concentration (%)")

    plt.ylabel("Microwave Power (W)")

    plt.title(f"{response} Contour Plot")

    plt.colorbar(contour,label=response)

    plt.tight_layout()

    plt.savefig(f"figures/{response}_Contour.png")

    plt.close()

print("All contour plots generated.")