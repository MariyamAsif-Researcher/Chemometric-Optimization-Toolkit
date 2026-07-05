import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from statsmodels.formula.api import ols

data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC","TFC","DPPH","ABTS","MCA"]

for response in responses:

    formula = f"{response} ~ DES + Power + Time + I(DES**2) + I(Power**2) + I(Time**2) + DES:Power + DES:Time + Power:Time"

    model = ols(formula,data=data).fit()

    DES = np.linspace(60,90,80)

    Power = np.linspace(220,440,80)

    DES_grid,Power_grid=np.meshgrid(DES,Power)

    Time=np.full_like(DES_grid,30)

    prediction=pd.DataFrame({

        "DES":DES_grid.ravel(),

        "Power":Power_grid.ravel(),

        "Time":Time.ravel()

    })

    Z=model.predict(prediction)

    Z=Z.values.reshape(DES_grid.shape)

    fig=plt.figure(figsize=(10,7))

    ax=fig.add_subplot(111,projection="3d")

    ax.plot_surface(

        DES_grid,

        Power_grid,

        Z,

        cmap="viridis",

        edgecolor="none"

    )

    ax.set_xlabel("DES (%)")

    ax.set_ylabel("Power (W)")

    ax.set_zlabel(response)

    ax.set_title(f"{response} Response Surface")

    plt.tight_layout()

    plt.savefig(f"figures/{response}_Surface.png")

    plt.close()

print("All surface plots generated.")