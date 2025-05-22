import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def create_features(data, target_column):
    df = data.copy()
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['minute'] = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)

    # Rolling features with multiple windows
    windows = [3, 6, 12, 24]
    for window in windows:

        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window, min_periods=1).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window, min_periods=1).std()
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window, min_periods=1).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window, min_periods=1).max()
        df[f'{target_column}_rolling_skew_{window}'] = df[target_column].rolling(window=window, min_periods=1).skew()
        df[f'{target_column}_rolling_roc_{window}'] = df[target_column].pct_change(periods=window)
    
    # Interaction features
    df['hour_target_interaction'] = df['hour'] * df[target_column]
    
    return df.bfill().ffill()

class TimeSeriesForecaster:
    def __init__(self, data=None, target_column=None, start_datetime=None, end_datetime=None):
        self.data = data
        self.target_column = target_column
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.arima_model = None
        self.fitted_arima = None
        
    def preprocess_data(self, data=None, target_column=None, start_datetime=None, end_datetime=None):
        if data is not None:
            self.data = data
        if target_column is not None:
            self.target_column = target_column
        if start_datetime is not None:
            self.start_datetime = start_datetime
        if end_datetime is not None:
            self.end_datetime = end_datetime
            
        if self.data is None or self.target_column is None:
            raise ValueError("Data and target column must be provided")
            
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
            
        # Sort by datetime
        self.data = self.data.sort_index()
        
        # Filter data based on start_datetime and end_datetime if provided
        if self.start_datetime is not None:
            self.data = self.data[self.data.index >= self.start_datetime]
        if self.end_datetime is not None:
            self.data = self.data[self.data.index <= self.end_datetime]
        
        # Create features
        self.data = create_features(self.data, self.target_column)
        
        # Handle missing values
        self.data = self.data.ffill().bfill()
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.ffill().bfill()
        
        # Verify no NaN or inf values remain
        if self.data.isna().any().any() or np.isinf(self.data.select_dtypes(include=np.number)).any().any():
            raise ValueError("Data still contains NaN or inf values after preprocessing")
        
        return self.data
    
    def train_models(self, order=(1,1,1)):
        if self.data is None or self.target_column is None:
            raise ValueError("Data must be preprocessed before training")
            
        self.arima_model = ARIMA(
            self.data[self.target_column],
            order=order,
            exog=self.data.drop(columns=[self.target_column])
        )
        self.fitted_arima = self.arima_model.fit()
        return self.fitted_arima
    
    def predict_next_hour(self, steps=1):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before making predictions")
        exog = self.data.drop(columns=[self.target_column]).iloc[-steps:]
        return self.fitted_arima.forecast(steps=steps, exog=exog)
    
    def evaluate_models(self, test_data):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before evaluation")
            
        predictions = self.predict_next_hour(steps=len(test_data))
        actual = test_data[self.target_column]
        
        metrics = {
            'ARIMA': {
                'MSE': mean_squared_error(actual, predictions),
                'MAE': mean_absolute_error(actual, predictions),
                'RMSE': np.sqrt(mean_squared_error(actual, predictions))
            }
        }
        
        metrics_df = pd.DataFrame(metrics).T
        os.makedirs('./results', exist_ok=True)
        metrics_df.to_csv('./results/model_metrics.csv')
        print("\nModel Metrics\n", metrics_df)
        return metrics_df
    
    def plot_forecast(self, test_data=None, forecast_steps=24):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before plotting")
            
        plt.figure(figsize=(15, 7))
        plt.plot(self.data.index, self.data[self.target_column], label='Historical Data', color='blue')
        
        forecast = self.predict_next_hour(steps=forecast_steps)
        forecast_index = pd.date_range(start=self.data.index[-1] + timedelta(hours=1), periods=forecast_steps, freq='h')
        plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
        
        if test_data is not None:
            plt.plot(test_data.index, test_data[self.target_column], label='Test Data', color='purple')
        
        plt.title('ARIMA Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig("./results/forecast.png")
        plt.close()

    def save_load_model(self, arima_path='./model/saved_models/arima_model.pkl', save=True):
        if save:
            if self.fitted_arima is None:
                raise ValueError("No model to save")
            os.makedirs(os.path.dirname(arima_path), exist_ok=True)
            with open(arima_path, 'wb') as f:
                pickle.dump({'fitted_model': self.fitted_arima, 'target_column': self.target_column}, f)
            print(f"ARIMA model saved to {arima_path}")
        else:
            if not os.path.exists(arima_path):
                raise FileNotFoundError("Model file not found")
            with open(arima_path, 'rb') as f:
                arima_data = pickle.load(f)
                self.fitted_arima = arima_data['fitted_model']
                self.target_column = arima_data['target_column']
            print("ARIMA model loaded successfully")

# Example usage:
if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('./dataset/clean_data.csv')
    
    # Convert Convert_time to datetime
    df['Convert_time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    
    # Set default datetime constraints
    default_start = pd.to_datetime('2022-07-04 05:43:37')
    default_end = pd.to_datetime('2022-07-22 13:25:41')
    
    # Get user input for datetime range
    try:
        start_datetime = pd.to_datetime(input("\nEnter start datetime (YYYY-MM-DD HH:MM:SS) or press Enter to use default: ") or default_start)
        end_datetime = pd.to_datetime(input("Enter end datetime (YYYY-MM-DD HH:MM:SS) or press Enter to use default: ") or default_end)
        
        # Validate datetime constraints
        if start_datetime < default_start or start_datetime > default_end:
            print(f"Start datetime must be after {default_start} and before {default_end}")
            exit()
        if end_datetime > default_end or end_datetime < default_start:
            print(f"End datetime must be before {default_end} and after {default_start}")
            exit()
        if (end_datetime - start_datetime).total_seconds() / 3600 < 24:
            print("Please provide at least 24 hours of data")
            exit()
            
    except ValueError:
        print("Invalid datetime format")
        exit()
    
    # Aggregate data by hour and calculate mean for all relevant columns
    hourly_data = df.set_index('Convert_time').resample('h').agg({
        'total_throughput': 'mean',
        'average_latency': 'mean'
    }).bfill().ffill()
    
    # Handle missing values
    if hourly_data.isna().any().any():
        hourly_data = hourly_data.fillna(hourly_data.mean())
    
    # Create DataFrame with hourly data
    sample_data = pd.DataFrame(hourly_data)
    
    try:
        # Initialize and train the model
        forecaster = TimeSeriesForecaster(sample_data, 'total_throughput', start_datetime, end_datetime)
        forecaster.preprocess_data()
        
        # Continue with model training (will automatically find best parameters)
        forecaster.train_models()
        
        # Save the trained model
        forecaster.save_load_model()
        
        # Get user input for prediction period
        try:
            prediction_hours = int(input("\nEnter prediction hours: "))
            if prediction_hours <= 0:
                print("Prediction period must be greater than 0")
                exit()
        except ValueError:
            print("Please enter a valid number of hours")
            exit()
        
        # Make predictions for the specified number of hours
        predictions = forecaster.predict_next_hour(steps=prediction_hours)
        print(f"\nPredictions for the next hours after the endtime:")
        
        for i in range(prediction_hours):
            next_hour = end_datetime + pd.Timedelta(hours=i+1)
            print(f"{next_hour.strftime('%Y-%m-%d %H:00')}: {predictions.iloc[i]:.2f}")
        
        # Plot results
        forecaster.plot_forecast(forecast_steps=prediction_hours)
        
        # Create a test set for evaluation (last 24 hours of data)
        test_data = sample_data.iloc[-24:].copy()
        train_data = sample_data.iloc[:-24].copy()
        
        # Evaluate the model
        forecaster.evaluate_models(test_data)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        exit()
