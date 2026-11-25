
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import load_train_data, add_rul_column

# Load data
df = load_train_data()
df = add_rul_column(df)

# Select sensor and operating setting features
feature_columns = [
    col for col in df.columns
    if col.startswith("sensor_") or col.startswith("setting_")
]

# Split by engine ID to avoid data leakage
engine_ids = df["engine_id"].unique()

train_engines, val_engines = train_test_split(
    engine_ids, test_size=0.2, random_state=42
)

train_data = df[df["engine_id"].isin(train_engines)]
val_data = df[df["engine_id"].isin(val_engines)]

X_train = train_data[feature_columns]
y_train = train_data["RUL"]

X_val = val_data[feature_columns]
y_val = val_data["RUL"]

# Models to compare
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


import joblib
from pathlib import Path

# Train the selected model on the complete training dataset
best_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
X = df[feature_columns]
y = df["RUL"]
best_model.fit(X, y)

# Save model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_columns
    },
    "models/best_model.pkl"
)

print("\nBest model saved to models/best_model.pkl")


# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    print(name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 3))
