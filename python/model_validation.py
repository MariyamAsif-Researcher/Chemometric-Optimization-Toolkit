import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

data = pd.read_csv("data/GCC_DEMO.csv")

responses = [
    "TPC",
    "TFC",
    "DPPH",
    "ABTS",
    "MCA"
]

results = []

# ---------------------------------------------------
# Loop through every response
# ---------------------------------------------------

for response in responses:

    print(f"\nRunning ANN for {response}")

    # -------------------------------
    # Independent variables
    # -------------------------------

    X = data[["DES", "Power", "Time"]]

    # -------------------------------
    # Response variable
    # -------------------------------

    y = data[[response]]

    # -------------------------------
    # Standardize X
    # -------------------------------

    X_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(X)

    # -------------------------------
    # Standardize y
    # -------------------------------

    y_scaler = StandardScaler()

    y_scaled = y_scaler.fit_transform(y).ravel()

    # -------------------------------
    # Train Test Split
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.20,
        random_state=42
    )

    # -------------------------------
    # ANN Model
    # -------------------------------

    model = MLPRegressor(

        hidden_layer_sizes=(20,10),

        activation="relu",

        solver="lbfgs",

        alpha=0.0001,

        max_iter=10000,

        random_state=42

    )

    # -------------------------------

    model.fit(X_train, y_train)

    prediction_scaled = model.predict(X_test)

    # -------------------------------
    # Convert back to original units
    # -------------------------------

    prediction = y_scaler.inverse_transform(
        prediction_scaled.reshape(-1,1)
    ).ravel()

    y_actual = y_scaler.inverse_transform(
        y_test.reshape(-1,1)
    ).ravel()

    # -------------------------------
    # Metrics
    # -------------------------------

    R2 = r2_score(y_actual, prediction)

    RMSE = np.sqrt(
        mean_squared_error(
            y_actual,
            prediction
        )
    )

    MAE = mean_absolute_error(
        y_actual,
        prediction
    )

    MSE = mean_squared_error(
        y_actual,
        prediction
    )

    results.append([
        response,
        R2,
        RMSE,
        MAE,
        MSE
    ])

    # ===================================================
    # Predicted vs Experimental
    # ===================================================

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_actual,
        prediction,
        s=80
    )

    minimum = min(
        y_actual.min(),
        prediction.min()
    )

    maximum = max(
        y_actual.max(),
        prediction.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        '--'
    )

    plt.xlabel("Experimental")

    plt.ylabel("Predicted")

    plt.title(f"{response}: Predicted vs Experimental")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/Predicted_vs_Experimental_{response}.png",
        dpi=300
    )

    plt.close()

    # ===================================================
    # Residual Plot
    # ===================================================

    residuals = y_actual - prediction

    plt.figure(figsize=(6,5))

    plt.scatter(
        prediction,
        residuals,
        s=80
    )

    plt.axhline(
        y=0,
        linestyle='--'
    )

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title(f"{response}: Residual Plot")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/Residuals_{response}.png",
        dpi=300
    )

    plt.close()

    # ===================================================
    # Residual Histogram
    # ===================================================

    plt.figure(figsize=(6,5))

    plt.hist(
        residuals,
        bins=8
    )

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.title(f"{response}: Residual Histogram")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"results/Residual_Histogram_{response}.png",
        dpi=300
    )

    plt.close()

# ---------------------------------------------------
# Save Results
# ---------------------------------------------------

performance = pd.DataFrame(

    results,

    columns=[
        "Response",
        "R2",
        "RMSE",
        "MAE",
        "MSE"
    ]

)

performance.to_csv(
    "results/Model_Performance.csv",
    index=False
)

print("\n==============================")
print("ANN Performance Summary")
print("==============================")

print(performance)

print("\nResults saved successfully!")