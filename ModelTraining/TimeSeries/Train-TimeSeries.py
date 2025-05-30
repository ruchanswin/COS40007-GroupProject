import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import joblib
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error


# === Constants ===
DATA_PATH = './ProcessedData/clean_data.csv'
MODEL_SAVE_PATH = './TrainedModel/TimeSeries/arima_model.pkl'
TEST_SIZE = 24
VAL_SIZE = 24
ARIMA_PARAM_GRID = list(product(range(3), repeat=3))  # Try (0–2) for p, d, q

# === Feature Engineering Function ===
def create_features(df, target_col='total_throughput'):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['minute'] = df.index.minute

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Rolling stats
    for w in [3, 6, 12, 24]:
        df[f'{target_col}_mean_{w}'] = df[target_col].rolling(window=w, min_periods=1).mean()
        df[f'{target_col}_std_{w}'] = df[target_col].rolling(window=w, min_periods=1).std()
        df[f'{target_col}_min_{w}'] = df[target_col].rolling(window=w, min_periods=1).min()
        df[f'{target_col}_max_{w}'] = df[target_col].rolling(window=w, min_periods=1).max()
        df[f'{target_col}_skew_{w}'] = df[target_col].rolling(window=w, min_periods=1).skew()
        df[f'{target_col}_roc_{w}'] = df[target_col].pct_change(periods=w)

    # Interaction
    df['hour_interaction'] = df['hour'] * df[target_col]

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    return df

# === Load and preprocess raw data ===
df = pd.read_csv(DATA_PATH)
df['Convert_time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df.set_index('Convert_time', inplace=True)

# === Resample and feature engineering ===
hourly_data = df.resample('h').agg({'total_throughput': 'mean'}).bfill().ffill()
data = create_features(hourly_data)

# === Train/Val/Test split ===
if len(data) < TEST_SIZE + VAL_SIZE + 1:
    raise ValueError("Not enough data for ARIMA training")

train_data = data[:-TEST_SIZE - VAL_SIZE]
val_data = data[-TEST_SIZE - VAL_SIZE:-TEST_SIZE]
test_data = data[-TEST_SIZE:]

# === Select exogenous features ===
# Only include time-based features that can be recreated during future forecasting
time_exog_cols = ['hour', 'day_of_week', 'minute',
                  'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin', 'minute_cos']

# Filter to what's available in data (safe subset)
exog_cols = [col for col in time_exog_cols if col in data.columns]

# === Grid Search for ARIMA(p,d,q) ===
all_results = []
best_aic = float('inf')
best_order = None
fitted_model = None

print("\nSearching best ARIMA(p,d,q) order and evaluating each model...")
for order in ARIMA_PARAM_GRID:
    try:
        model = ARIMA(train_data['total_throughput'], order=order, exog=train_data[exog_cols])
        fitted = model.fit()
        aic = fitted.aic

        # Forecast and evaluate on test set
        forecast = fitted.forecast(steps=TEST_SIZE, exog=test_data[exog_cols])
        y_true = test_data['total_throughput'].values
        y_pred = forecast.values

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        all_results.append({
            'order': order,
            'aic': aic,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'model': fitted
        })

        print(f"ARIMA{order} | AIC: {aic:.2f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        # Track best model by AIC
        if aic < best_aic:
            best_aic = aic
            best_order = order
            fitted_model = fitted

    except Exception as e:
        print(f"Failed ARIMA{order}: {e}")
        continue

if fitted_model is None:
    raise RuntimeError("All ARIMA configurations failed.")

df_results = pd.DataFrame([
    {k: v for k, v in res.items() if k != 'model'}
    for res in all_results
])
df_results.to_csv("./TrainedModel/TimeSeries/ARIMA_model_metrics.csv", index=False)
print("All models result saved.")
# === Save model ===
last_train_timestamp = train_data.index.max()
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump({
    'fitted_model': fitted_model,
    'target_column': 'total_throughput',
    'last_train_timestamp': last_train_timestamp,
    'best_order': best_order,
    'features': exog_cols
}, MODEL_SAVE_PATH)

print(f"\nBest ARIMA Model: ARIMA{best_order} (lowest AIC = {best_aic:.2f}) saved to: {MODEL_SAVE_PATH}")

# === Forecast on test set ===
print("\nGenerating forecast for test set...")
forecast = fitted_model.forecast(steps=TEST_SIZE, exog=test_data[exog_cols])

# === Evaluation ===
y_true = test_data['total_throughput'].values
y_pred = forecast.values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print(f"\nBest ARIMA Model: ARIMA{best_order}")
print("Evaluation Metrics on Test Set:")
print(f" - MSE  : {mse:.4f}")
print(f" - RMSE : {rmse:.4f}")
print(f" - MAE  : {mae:.4f}")
