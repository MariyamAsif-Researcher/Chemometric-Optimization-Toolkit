import pandas as pd
import numpy as np

from scipy.optimize import minimize

import statsmodels.formula.api as smf


# ---------------------------------------
# Load Dataset
# ---------------------------------------

data = pd.read_csv("data/GCC_DEMO.csv")


responses = ["TPC","TFC","DPPH","ABTS","MCA"]

results = []


for response in responses:

    formula = f"{response} ~ DES + Power + Time + I(DES**2) + I(Power**2) + I(Time**2) + DES:Power + DES:Time + Power:Time"

    model = smf.ols(formula,data=data).fit()


    # Prediction Function

    def predict(x):

        df = pd.DataFrame({

            "DES":[x[0]],
            "Power":[x[1]],
            "Time":[x[2]]

        })

        return -model.predict(df)[0]


    bounds = [

        (60,90),      # DES

        (220,440),    # Power

        (20,40)       # Time

    ]


    start = [75,330,30]


    optimum = minimize(

        predict,

        start,

        bounds=bounds

    )


    DES_opt = optimum.x[0]
    Power_opt = optimum.x[1]
    Time_opt = optimum.x[2]

    Max_Response = -optimum.fun


    results.append([

        response,

        DES_opt,

        Power_opt,

        Time_opt,

        Max_Response

    ])


results = pd.DataFrame(

    results,

    columns=[

        "Response",

        "Optimal_DES",

        "Optimal_Power",

        "Optimal_Time",

        "Predicted_Response"

    ]

)

print(results)

results.to_csv("results/Optimization_Results.csv",index=False)

print("\nOptimization Complete!")