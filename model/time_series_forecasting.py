import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta
import pickle
import os

class TimeSeriesForecaster:
    def __init__(self, data=None, target_column=None):
        self.data = data
        self.target_column = target_column
        self.arima_model = None
        self.fitted_arima = None
        
    def preprocess_data(self, data=None, target_column=None):
        if data is not None:
            self.data = data
        if target_column is not None:
            self.target_column = target_column
            
        if self.data is None or self.target_column is None:
            raise ValueError("Data and target column must be provided")
            
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
            
        # Sort by datetime
        self.data = self.data.sort_index()
        
        # Handle missing values
        self.data = self.data.ffill()
        
        return self.data
    
    def train_models(self, order=(1, 1, 1)):
        if self.data is None or self.target_column is None:
            raise ValueError("Data must be preprocessed before training")
            
        # Train ARIMA model
        self.arima_model = ARIMA(
            self.data[self.target_column],
            order=order
        )
        
        self.fitted_arima = self.arima_model.fit()
        
        return self.fitted_arima
    
    def predict_next_hour(self, steps=1):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before making predictions")
            
        predictions = self.fitted_arima.forecast(steps=steps)
        return predictions
    
    def evaluate_models(self, test_data):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before evaluation")
            
        predictions = self.predict_next_hour(steps=len(test_data))
        actual = test_data[self.target_column]
        
        # Calculate various metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'ARIMA': {
                'MSE': f"{mse:.4f}",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}"
            }
        }
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(metrics).T
        
        # Create table visualization
        plt.figure(figsize=(12, 4))
        plt.axis('off')  # Hide axes
        
        # Create table
        table = plt.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2'] * len(metrics_df.columns),
            cellColours=[['#ffffff'] * len(metrics_df.columns) for _ in range(len(metrics_df))]
        )
        
        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title('ARIMA Model Metrics', pad=20, fontsize=14)
        
        # Save the plot
        plt.savefig('./results/model_metrics.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close()
        
        # Print the metrics table
        print("\nModel Metrics")
        print(metrics_df)
        
        return metrics_df
    
    def plot_forecast(self, test_data=None, forecast_steps=24):
        if self.fitted_arima is None:
            raise ValueError("Model must be trained before plotting")
            
        plt.figure(figsize=(15, 7))
        
        # Plot historical data
        plt.plot(self.data.index, self.data[self.target_column], label='Historical Data', color='blue')
        
        # Generate forecasts
        forecast = self.predict_next_hour(steps=forecast_steps)
        forecast_index = pd.date_range(
            start=self.data.index[-1] + timedelta(hours=1),
            periods=forecast_steps,
            freq='h'
        )
        
        # Add the last historical point to the forecast for continuity
        forecast = np.concatenate([[self.data[self.target_column].iloc[-1]], forecast])
        forecast_index = pd.date_range(
            start=self.data.index[-1],
            periods=forecast_steps + 1,
            freq='h'
        )
        
        # Plot forecast
        plt.plot(forecast_index, forecast, label='ARIMA Forecast', 
                color='red', linestyle='--')
        
        # Plot test data if provided
        if test_data is not None:
            plt.plot(test_data.index, test_data[self.target_column], 
                    label='Test Data', color='purple')
        
        plt.title('ARIMA Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig("./results/forecast.png")
        plt.close()

    def save_models(self, arima_path='./model/saved_models/arima_model.pkl'):
        """Save the trained model to file"""
        if self.fitted_arima is None:
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(arima_path), exist_ok=True)
        
        # Save ARIMA model
        with open(arima_path, 'wb') as f:
            pickle.dump({
                'fitted_model': self.fitted_arima,
                'target_column': self.target_column
            }, f)
        print(f"ARIMA model saved to {arima_path}")
        
    def load_models(self, arima_path='./model/saved_models/arima_model.pkl'):
        """Load trained model from file"""
        if not os.path.exists(arima_path):
            raise FileNotFoundError("Model file not found")
            
        # Load ARIMA model
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
    
    # Define valid date range
    min_date = pd.to_datetime('2022-07-04 05:00:00')
    max_date = pd.to_datetime('2022-07-22 14:00:00')
    
    # Get user input for the date range
    start_date_str = input("Enter the starttime (YYYY-MM-DD HH:MM): ")
    end_date_str = input("Enter the endtime (YYYY-MM-DD HH:MM): ")
    
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        if end_date < start_date:
            print("End date must be after start date")
            exit()
            
        if start_date < min_date:
            print(f"Start date must be after {min_date.strftime('%Y-%m-%d %H:%M')}")
            exit()
            
        if end_date > max_date:
            print(f"End date must be before {max_date.strftime('%Y-%m-%d %H:%M')}")
            exit()
            
    except ValueError:
        print("Invalid datetime format. Please use YYYY-MM-DD HH:MM format.")
        exit()
    
    # Filter data for the date range
    mask = (df['Convert_time'] >= start_date) & (df['Convert_time'] <= end_date)
    range_data = df[mask].copy()
    
    if len(range_data) == 0:
        print(f"No data found for the specified date range")
        exit()
    
    # Aggregate data by hour and calculate mean total_throughput
    hourly_data = range_data.set_index('Convert_time').resample('h')['total_throughput'].mean()
    
    # Handle missing values by forward fill and then backward fill
    hourly_data = hourly_data.bfill().ffill()
    
    # If there are still any NaN values, fill with the mean
    if hourly_data.isna().any():
        hourly_data = hourly_data.fillna(hourly_data.mean())
    
    # Create DataFrame with hourly data
    sample_data = pd.DataFrame({'value': hourly_data})
    
    # Initialize and train the model
    forecaster = TimeSeriesForecaster(sample_data, 'value')
    forecaster.preprocess_data()
    forecaster.train_models()
    
    # Save the trained model
    forecaster.save_models()
    
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
        next_hour = end_date + pd.Timedelta(hours=i+1)
        print(f"{next_hour.strftime('%Y-%m-%d %H:00')}: {predictions.iloc[i]:.2f}")
    
    # Plot results
    forecaster.plot_forecast(forecast_steps=prediction_hours)
    
    # Create a test set for evaluation (last 24 hours of data)
    test_data = sample_data.iloc[-24:].copy()
    train_data = sample_data.iloc[:-24].copy()
    
    # Evaluate the model
    forecaster.evaluate_models(test_data)
