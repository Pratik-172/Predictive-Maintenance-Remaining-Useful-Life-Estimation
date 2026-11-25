
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import COLUMN_NAMES


# Load saved model
saved = joblib.load("models/best_model.pkl")

model = saved["model"]
feature_columns = saved["feature_columns"]


# Load official test data
test_data = pd.read_csv(
    "data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)


# Load actual RUL values for test engines
actual_rul = pd.read_csv(
    "data/raw/RUL_FD001.txt",
    sep=r"\s+",
    header=None
)[0]


# Select the final available cycle for each test engine
last_rows = (
    test_data.groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
)


# Prepare features
X_test = last_rows[feature_columns]


# Predict RUL
predictions = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(actual_rul, predictions)
rmse = mean_squared_error(actual_rul, predictions) ** 0.5
r2 = r2_score(actual_rul, predictions)


print("Official FD001 Test Results")
print("--------------------------")
print("Test engines:", len(actual_rul))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Show sample predictions
results = pd.DataFrame({
    "engine_id": last_rows["engine_id"].values,
    "actual_RUL": actual_rul.values,
    "predicted_RUL": predictions.round(2)
})

print("\nSample predictions:")
print(results.head(10))


import matplotlib.pyplot as plt
from pathlib import Path

# Create results folder
Path("results").mkdir(exist_ok=True)

# Save predictions
results.to_csv("results/test_predictions.csv", index=False)

# Calculate prediction errors
errors = results["predicted_RUL"] - results["actual_RUL"]

# 1. Actual vs Predicted RUL
plt.figure(figsize=(8, 6))
plt.scatter(results["actual_RUL"], results["predicted_RUL"])
plt.plot([0, 130], [0, 130], linestyle="--")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.savefig("results/actual_vs_predicted.png")
plt.close()

# 2. Prediction Error Distribution
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20)
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Engines")
plt.title("Prediction Error Distribution")
plt.savefig("results/error_distribution.png")
plt.close()

# 3. Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
importance.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Important Features")
plt.savefig("results/feature_importance.png")
plt.close()

print("\nGraphs and predictions saved in results/")
