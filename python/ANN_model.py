import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data/GCC_DEMO.csv")

responses = [

"TPC",

"TFC",

"DPPH",

"ABTS",

"MCA"

]

results=[]

for response in responses:

    X=data[["DES","Power","Time"]]

    y=data[response]

    X_train,X_test,y_train,y_test=train_test_split(

        X,

        y,

        test_size=0.25,

        random_state=42

    )

    model=MLPRegressor(

        hidden_layer_sizes=(10,),

        activation="relu",

        solver="adam",

        max_iter=5000,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    prediction=model.predict(X_test)

    R2=r2_score(

        y_test,

        prediction

    )

    RMSE=np.sqrt(

        mean_squared_error(

            y_test,

            prediction

        )

    )

    results.append([

        response,

        R2,

        RMSE

    ])

results=pd.DataFrame(

    results,

    columns=[

        "Response",

        "R2",

        "RMSE"

    ]

)

print(results)

results.to_csv(

    "results/ANN_Performance.csv",

    index=False

)