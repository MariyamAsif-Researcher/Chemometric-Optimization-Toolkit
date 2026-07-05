import pandas as pd
import numpy as np

from scipy.optimize import minimize

import statsmodels.formula.api as smf

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

data = pd.read_csv("data/GCC_DEMO.csv")

responses = ["TPC","TFC","DPPH","ABTS","MCA"]

models = {}

minimum = {}
maximum = {}

# ---------------------------------------------------
# Fit one RSM model for every response
# ---------------------------------------------------

for response in responses:

    formula = f"{response} ~ DES + Power + Time + I(DES**2) + I(Power**2) + I(Time**2) + DES:Power + DES:Time + Power:Time"

    model = smf.ols(formula,data=data).fit()

    models[response] = model

    minimum[response] = data[response].min()

    maximum[response] = data[response].max()

# ---------------------------------------------------
# Desirability Function
# ---------------------------------------------------

def overall_desirability(x):

    df = pd.DataFrame({

        "DES":[x[0]],

        "Power":[x[1]],

        "Time":[x[2]]

    })

    desirabilities = []

    for response in responses:

        prediction = models[response].predict(df)[0]

        d = (prediction - minimum[response]) / (maximum[response] - minimum[response])

        d = max(0,min(1,d))

        desirabilities.append(d)

    D = np.prod(desirabilities)**(1/len(desirabilities))

    return -D

# ---------------------------------------------------
# Optimization
# ---------------------------------------------------

bounds = [

    (60,90),

    (220,440),

    (20,40)

]

start = [75,330,30]

result = minimize(

    overall_desirability,

    start,

    bounds=bounds

)

# ---------------------------------------------------
# Output
# ---------------------------------------------------

best = result.x

print("\nOptimal Conditions")

print("---------------------------")

print("DES:",round(best[0],2))

print("Power:",round(best[1],2))

print("Time:",round(best[2],2))

print()

df = pd.DataFrame({

    "DES":[best[0]],

    "Power":[best[1]],

    "Time":[best[2]]

})

print("Predicted Responses")

print("---------------------------")

output=[]

for response in responses:

    value=models[response].predict(df)[0]

    print(response,":",round(value,3))

    output.append(value)

print()

print("Overall Desirability =",round(-result.fun,4))

results=pd.DataFrame({

    "Response":responses,

    "Prediction":output

})

results.to_csv("results/Desirability_Optimization.csv",index=False)

print("\nResults saved successfully.")