
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data_utils import COLUMN_NAMES

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("Predictive Maintenance")
st.subheader("Remaining Useful Life (RUL) Estimation")

# Load model
saved = joblib.load("models/best_model.pkl")
model = saved["model"]
feature_columns = saved["feature_columns"]

# Load test data
test_data = pd.read_csv(
    "data/raw/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES
)

# Load actual RUL values
actual_rul = pd.read_csv(
    "data/raw/RUL_FD001.txt",
    sep=r"\s+",
    header=None
)[0]

# Get final observation for each engine
last_rows = (
    test_data.groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
    .reset_index(drop=True)
)

# Predict RUL for test engines
predictions = model.predict(last_rows[feature_columns])

results = pd.DataFrame({
    "engine_id": last_rows["engine_id"],
    "actual_RUL": actual_rul,
    "predicted_RUL": predictions.round(2)
})

# Engine selection
engine_id = st.selectbox(
    "Select Engine ID",
    results["engine_id"].tolist()
)

selected = results[results["engine_id"] == engine_id].iloc[0]
sensor_data = last_rows[last_rows["engine_id"] == engine_id].iloc[0]

# Display prediction
col1, col2, col3 = st.columns(3)

col1.metric(
    "Predicted RUL",
    f"{selected['predicted_RUL']:.1f} cycles"
)

col2.metric(
    "Actual RUL",
    f"{selected['actual_RUL']} cycles"
)

col3.metric(
    "Prediction Error",
    f"{abs(selected['predicted_RUL'] - selected['actual_RUL']):.1f} cycles"
)

st.divider()

# Sensor readings
st.subheader("Engine Sensor Readings")

# Sort sensor names numerically
sensor_columns = sorted(
    [col for col in feature_columns if col.startswith("sensor_")],
    key=lambda col: int(col.split("_")[1])
)

sensor_values = pd.DataFrame({
    "Sensor": sensor_columns,
    "Reading": [sensor_data[col] for col in sensor_columns]
})

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    sensor_values["Sensor"],
    sensor_values["Reading"]
)

ax.set_title("Engine Sensor Readings")
ax.set_xlabel("Sensor Value")
ax.set_ylabel("Sensor")
ax.invert_yaxis()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# Model performance
st.subheader("Model Performance")

st.write("Official FD001 test-set performance:")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "13.12 cycles")
col2.metric("RMSE", "18.36 cycles")
col3.metric("R² Score", "0.805")

st.caption(
    "Dataset: NASA C-MAPSS FD001. "
    "Predictions are generated using a Gradient Boosting model."
)
