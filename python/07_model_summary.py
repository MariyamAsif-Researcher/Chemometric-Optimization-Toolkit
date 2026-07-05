import pandas as pd
import numpy as np
import os

import statsmodels.api as sm
from statsmodels.formula.api import ols

# ===========================================
# Create Folder
# ===========================================

os.makedirs("results/model_summary", exist_ok=True)

# ===========================================
# Load Dataset
# ===========================================

data = pd.read_csv("data/GCC_DEMO.csv")

# ===========================================
# Responses
# ===========================================

responses = [
    "TPC",
    "TFC",
    "DPPH",
    "ABTS",
    "MCA"
]

summary_table = []

# ===========================================
# Loop
# ===========================================

for response in responses:

    formula = (
        f"{response} ~ "
        "DES + Power + Time + "
        "DES:Power + DES:Time + Power:Time + "
        "I(DES**2) + I(Power**2) + I(Time**2)"
    )

    model = ols(formula, data=data).fit()

    observed = data[response]

    predicted = model.fittedvalues

    residuals = observed - predicted

    rmse = np.sqrt(np.mean(residuals**2))

    press = np.sum(residuals**2)

    mean = observed.mean()

    cv = (rmse / mean) * 100

    summary_table.append({

        "Response": response,

        "R2": round(model.rsquared,4),

        "Adj_R2": round(model.rsquared_adj,4),

        "RMSE": round(rmse,4),

        "PRESS": round(press,4),

        "Mean": round(mean,4),

        "CV%": round(cv,3),

        "AIC": round(model.aic,2),

        "BIC": round(model.bic,2)

    })

summary = pd.DataFrame(summary_table)

print(summary)

summary.to_csv(
    "results/model_summary/model_statistics.csv",
    index=False
)

print("\nModel summary saved.")