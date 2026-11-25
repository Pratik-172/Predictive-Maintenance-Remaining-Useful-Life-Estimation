# Predictive Maintenance with Remaining Useful Life Estimation

A Machine Learning project that predicts the Remaining Useful Life (RUL) of turbofan engines using historical sensor data from the NASA C-MAPSS FD001 dataset.

The project compares multiple regression algorithms, evaluates the selected model on the official FD001 test dataset, and provides an interactive Streamlit dashboard for exploring engine-level predictions and sensor readings.

---

## 1. Problem Statement

Unexpected machine failures can lead to equipment downtime, maintenance costs and operational losses.

Predictive maintenance uses machine condition data to estimate when maintenance may be required, helping maintenance teams plan inspections and replacements before a failure occurs.

The objective of this project is to develop a machine learning model that estimates the **Remaining Useful Life (RUL)** of a turbofan engine from its operating conditions and sensor measurements.

RUL is measured in operating cycles.

For example, an RUL prediction of 25 cycles means that the model estimates approximately 25 operating cycles remaining before the engine reaches its failure point in the dataset.

<img width="1308" height="2022" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/5e11869e-edc5-49fa-ac30-ac2db917fa78" />


---

## 2. Project Objectives

- Load and preprocess the NASA C-MAPSS FD001 dataset.
- Generate Remaining Useful Life labels for training data.
- Compare multiple regression algorithms.
- Select a suitable model using validation performance.
- Evaluate the selected model on the official test dataset.
- Visualize actual and predicted RUL values.
- Provide an interactive dashboard for engine-level prediction analysis.

---

## 3. Dataset

### NASA C-MAPSS Turbofan Engine Dataset

This project uses the **FD001 subset** of the NASA C-MAPSS turbofan engine degradation simulation dataset.

The dataset contains multivariate time-series records representing the operating history of simulated turbofan engines.

Each engine is represented by multiple operating cycles. Sensor readings change as the engine operates and gradually degrades.

### Dataset Files

The following files are used:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

| Dataset | Records | Engines |
|---|---:|---:|
| Training data | 20,631 | 100 |
| Test data | 13,096 | 100 |
| Official test RUL labels | 100 | 100 |

### Features

Each telemetry record contains 26 columns:

- Engine ID
- Operating cycle
- 3 operational settings
- 21 sensor measurements

The model uses the operational settings and sensor measurements as input features.

Engine ID and operating cycle are not used as direct model input features.

The sensor channels are retained using their original anonymized names, such as `sensor_1`, `sensor_2`, and `sensor_21`.

### Dataset Source

NASA C-MAPSS dataset:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

The dataset is provided by NASA's Prognostics Center of Excellence.

---

## 4. Remaining Useful Life Calculation

For the training dataset, the maximum operating cycle of each engine is calculated.

The RUL for an individual record is calculated as:

```text
RUL = Maximum cycle of that engine - Current cycle
```

For example, if an engine's maximum training cycle is 192:

| Current Cycle | RUL |
|---:|---:|
| 188 | 4 |
| 189 | 3 |
| 190 | 2 |
| 191 | 1 |
| 192 | 0 |

### RUL Capping

The training RUL values are capped at 125 cycles.

```python
RUL = (max_cycle - current_cycle).clip(upper=125)
```

This means that RUL values greater than 125 are represented as 125 during training.

The model therefore learns to predict RUL within the capped target range.

---

## 5. Data Preprocessing

The preprocessing pipeline includes:

1. Loading the whitespace-separated dataset files using Pandas.
2. Assigning column names to engine ID, cycle, operational settings and sensor measurements.
3. Calculating the maximum operating cycle for each training engine.
4. Generating the RUL target.
5. Capping training RUL at 125 cycles.
6. Selecting operational settings and sensor measurements as model features.

The final model uses 24 input features:

- 3 operational settings
- 21 sensor measurements

---

## 6. Model Development

Three regression algorithms were implemented and compared.

### Linear Regression

Linear Regression is used as a baseline model to estimate RUL using a linear relationship between the input features and the target.

### Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a regression prediction.

The implemented model uses 100 estimators.

### Gradient Boosting Regressor

Gradient Boosting builds an ensemble of decision trees sequentially. Each new tree attempts to improve the predictions made by the previous ensemble.

The implemented model uses 100 estimators.

### Model Selection

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The model with the best validation performance was selected.

Gradient Boosting achieved the lowest validation RMSE and was selected for final training.

---

## 7. Validation Results

The training dataset was divided into training and validation subsets using engine IDs.

Approximately 80% of the engines were used for training and 20% for validation, with `random_state=42`.

The split was performed by engine rather than randomly by individual rows. This prevents records from the same engine appearing in both training and validation subsets.

### Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 16.48 | 20.41 | 0.761 |
| Random Forest | 12.59 | 17.24 | 0.829 |
| Gradient Boosting | 12.58 | 17.12 | 0.832 |

### Validation Interpretation

Gradient Boosting achieved the best validation results:

- MAE: 12.58 cycles
- RMSE: 17.12 cycles
- R²: 0.832

Random Forest performed very similarly to Gradient Boosting.

The difference between the two models was small, but Gradient Boosting was selected based on its slightly better validation metrics.

After model selection, the selected Gradient Boosting model was retrained using the complete training dataset.

The trained model and its feature-column information were saved to:

```text
models/best_model.pkl
```

---

## 8. Official Test Results

The selected model was evaluated using the official NASA FD001 test dataset.

For each test engine, the final available operating-cycle record was selected.

The model predicted the RUL at that final observation and compared it against the corresponding official RUL value provided in `RUL_FD001.txt`.

### Final Test Performance

| Metric | Result |
|---|---:|
| Test engines | 100 |
| MAE | 13.12 cycles |
| RMSE | 18.36 cycles |
| R² Score | 0.805 |

### Test Result Interpretation

**MAE = 13.12 cycles**

On average, the absolute difference between the predicted and actual RUL was approximately 13.12 operating cycles.

**RMSE = 18.36 cycles**

RMSE penalizes larger prediction errors more strongly than MAE. Its value indicates that some predictions have larger errors than the average absolute error.

**R² = 0.805**

The model achieved an R² score of 0.805 on the official test set.

This indicates that the model explains approximately 80.5% of the variation in the test RUL values relative to a mean-based baseline. R² is not the same as prediction accuracy percentage.

The official test results are separate from the validation results and are used to assess the selected model on the provided test engines.

---

## 9. Evaluation Visualizations

The evaluation script generates the following outputs:

```text
results/
├── test_predictions.csv
├── actual_vs_predicted.png
├── error_distribution.png
└── feature_importance.png
```

### Actual vs Predicted RUL

<img width="800" height="600" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/02b233b8-2077-4734-8759-ccd07a910989" />

This graph compares the actual RUL values with the model's predicted RUL values.

Predictions closer to the ideal comparison line indicate smaller prediction errors.

### Error Distribution

<img width="800" height="600" alt="error_distribution" src="https://github.com/user-attachments/assets/68670a79-c050-488d-a966-a06baa33411e" />

This graph shows how the prediction errors are distributed across the test engines.

It helps identify whether errors are generally small or whether the model has larger errors for some engines.

### Feature Importance

<img width="800" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/7f25a650-7d1a-4ff8-9df1-42e6f6b854ed" />

This graph displays the feature importance values provided by the trained Gradient Boosting model.

It helps identify which input features contributed more strongly to the model's predictions.

---

## 10. Streamlit Dashboard

<img width="1803" height="463" alt="Screenshot 2026-11-25 233919" src="https://github.com/user-attachments/assets/e07e6be5-07cb-4770-9b3e-19f71a7909ed" />
<img width="1786" height="567" alt="Screenshot 2026-11-25 233939" src="https://github.com/user-attachments/assets/637fd1af-5806-4643-b672-47b17fc4c31a" />
<img width="1803" height="576" alt="Screenshot 2026-11-25 233955" src="https://github.com/user-attachments/assets/7a9910dd-9535-4f38-bb80-b190bd4322da" />

The project includes an interactive Streamlit dashboard.

The dashboard provides:

- Engine ID selection
- Predicted RUL
- Actual RUL
- Prediction error
- Engine sensor readings
- Model performance metrics

The sensor chart displays the selected engine's sensor readings in numerical sensor order.

The dashboard loads the trained model and generates predictions for the final observation of each test engine.

**Important:** The current dashboard uses the official test dataset and displays actual RUL values for demonstration and evaluation. In a real deployment, the actual future RUL of an operating engine would not be known.

---

## 11. Project Structure

```text
Predictive-Maintenance-RUL/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   │
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── test_predictions.csv
│   ├── actual_vs_predicted.png
│   ├── error_distribution.png
│   └── feature_importance.png
│
├── screenshots/
│
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── train.py
    └── evaluate.py
```

---

## 12. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## 13. Installation and Execution

### Step 1: Open the project directory

```powershell
cd "C:\Users\osdoi\OneDrive\Desktop\Predictive-Maintenance-RUL"
```

### Step 2: Install the required libraries

```powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### Step 3: Train the models

```powershell
python src/train.py
```

This performs model comparison, selects the best model, retrains it using the full training dataset and saves the trained model.

### Step 4: Evaluate the trained model

```powershell
python src/evaluate.py
```

This evaluates the model on the official FD001 test dataset and generates the prediction CSV and evaluation graphs.

### Step 5: Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the following URL in a browser:

```text
http://localhost:8501
```

To stop the Streamlit application, press `Ctrl + C` in the terminal.

---

## 14. Limitations

- The project uses only the NASA C-MAPSS FD001 subset.
- The dataset is based on simulated turbofan engine degradation rather than live industrial machinery.
- Training RUL values are capped at 125 cycles.
- The model is evaluated using 100 official test engines.
- The model uses the final available test observation for each engine.
- Sensor channels are anonymized and do not directly identify physical sensor meanings.
- The current dashboard is an evaluation and demonstration interface, not a live industrial monitoring system.
- Prediction performance on this dataset does not guarantee the same performance on real machinery.

---

## 15. Future Scope

Possible future improvements include:

- Testing additional C-MAPSS subsets such as FD002, FD003 and FD004.
- Exploring time-series features based on sensor history.
- Comparing additional regression algorithms.
- Improving model performance through feature selection and hyperparameter tuning.
- Adding a user-input prediction mode.
- Developing a live sensor-data interface.
- Integrating real-time monitoring and maintenance alerts.

---

## 16. Conclusion

This project demonstrates an end-to-end machine learning workflow for Remaining Useful Life estimation.

The implementation includes dataset preprocessing, RUL target generation, comparison of three regression algorithms, engine-wise validation, final model training, official test evaluation and an interactive Streamlit dashboard.

The Gradient Boosting model achieved an official FD001 test MAE of 13.12 cycles, RMSE of 18.36 cycles and R² score of 0.805.

The project demonstrates how historical sensor measurements can be used to estimate the remaining operating life of simulated turbofan engines and support predictive maintenance planning.

---

## 17. Dataset Reference

A. Saxena, K. Goebel, D. Simon and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), 2008.

NASA C-MAPSS Dataset:

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
