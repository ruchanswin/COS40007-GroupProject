
# Train-TimeSeries.ipynb

This notebook trains an ARIMA model for time-series forecasting of 5G network performance (total throughput) using historical data.

---

## Location

`<Local Directory>/5G Zone Prediction System/ModelTraining/TimeSeries/Train-TimeSeries.ipynb`

---

## Purpose

To forecast hourly future throughput values using historical throughput trends and time-based features.

---

## Features

- `total_throughput` (target)
- Exogenous:
  - `hour` of the day
  - `day_of_week`

---

## Workflow

1. Load and parse `clean_data.csv`
2. Perform similar feature engineering like the clean_data_Training.csv and resample to hourly data
3. Generate time features
4. Split into train and test sets
5. Fit ARIMA model with external regressors
6. Evaluate with RMSE and MAE
7. Save the trained model using `pickle`

---

## Outputs

- `arima_model.pkl`: Trained ARIMA model
- Plots: Predicted vs Actual throughput

---

## Usage Notes

- Ensure timestamps are in the format `YYYY-MM-DD HH:MM:SS`
- Data before `2022-07-20 13:00:00` is used for training
