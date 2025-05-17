import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
        self.sarimax_model = None
        self.arima_model = None
        self.fitted_sarimax = None
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
    
    def train_models(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 24)):
        if self.data is None or self.target_column is None:
            raise ValueError("Data must be preprocessed before training")
            
        # Train SARIMAX model
        self.sarimax_model = SARIMAX(
            self.data[self.target_column],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_sarimax = self.sarimax_model.fit(
            disp=False,
            maxiter=100,
            method='lbfgs'
        )
        
        # Train ARIMA model
        self.arima_model = ARIMA(
            self.data[self.target_column],
            order=order
        )
        
        self.fitted_arima = self.arima_model.fit()
        
        return self.fitted_sarimax, self.fitted_arima
    
    def predict_next_hour(self, steps=1, model_type='both'):
        if self.fitted_sarimax is None or self.fitted_arima is None:
            raise ValueError("Models must be trained before making predictions")
            
        predictions = {}
        
        if model_type in ['sarimax', 'both']:
            predictions['SARIMAX'] = self.fitted_sarimax.forecast(steps=steps)
            
        if model_type in ['arima', 'both']:
            predictions['ARIMA'] = self.fitted_arima.forecast(steps=steps)
            
        return predictions
    
    def evaluate_models(self, test_data):
        if self.fitted_sarimax is None or self.fitted_arima is None:
            raise ValueError("Models must be trained before evaluation")
            
        predictions = self.predict_next_hour(steps=len(test_data))
        actual = test_data[self.target_column]
        
        metrics = {}
        for model_name, pred in predictions.items():
            # Calculate various metrics
            mse = mean_squared_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mse)
            
            metrics[model_name] = {
                'MSE': f"{mse:.4f}",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}"
            }
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(metrics).T
        
        # Create table visualization
        plt.figure(figsize=(12, 4))
        plt.axis('off')  # Hide axes
        
        # Create table
        table = plt.table(
            cellText=metrics_df.values,
            rowLabels=metrics_df.index,
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
        plt.title('Model Comparison Metrics', pad=20, fontsize=14)
        
        # Save the plot
        plt.savefig('./results/model_comparison_metrics.png', 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close()
        
        # Print the metrics table
        print("\nModel Comparison Metrics:")
        print(metrics_df)
        
        return metrics_df
    
    def plot_forecast(self, test_data=None, forecast_steps=24):
        if self.fitted_sarimax is None or self.fitted_arima is None:
            raise ValueError("Models must be trained before plotting")
            
        plt.figure(figsize=(15, 7))
        
        # Plot historical data
        plt.plot(self.data.index, self.data[self.target_column], label='Historical Data', color='blue')
        
        # Generate forecasts
        forecasts = self.predict_next_hour(steps=forecast_steps)
        forecast_index = pd.date_range(
            start=self.data.index[-1] + timedelta(hours=1),
            periods=forecast_steps,
            freq='h'
        )
        
        # Plot forecasts
        colors = {'SARIMAX': 'red', 'ARIMA': 'green'}
        for model_name, forecast in forecasts.items():
            plt.plot(forecast_index, forecast, label=f'{model_name} Forecast', 
                    color=colors[model_name], linestyle='--')
        
        # Plot test data if provided
        if test_data is not None:
            plt.plot(test_data.index, test_data[self.target_column], 
                    label='Test Data', color='purple')
        
        plt.title('Time Series Forecast Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig("./results/forecast_comparison.png")
        plt.close()

    def save_models(self, sarimax_path='./model/saved_models/sarimax_model.pkl',
                   arima_path='./model/saved_models/arima_model.pkl'):
        """Save the trained models to files"""
        if self.fitted_sarimax is None or self.fitted_arima is None:
            raise ValueError("No trained models to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sarimax_path), exist_ok=True)
        
        # Save SARIMAX model
        with open(sarimax_path, 'wb') as f:
            pickle.dump({
                'fitted_model': self.fitted_sarimax,
                'target_column': self.target_column
            }, f)
        print(f"SARIMAX model saved to {sarimax_path}")
        
        # Save ARIMA model
        with open(arima_path, 'wb') as f:
            pickle.dump({
                'fitted_model': self.fitted_arima,
                'target_column': self.target_column
            }, f)
        print(f"ARIMA model saved to {arima_path}")
        
    def load_models(self, sarimax_path='./model/saved_models/sarimax_model.pkl',
                   arima_path='./model/saved_models/arima_model.pkl'):
        """Load trained models from files"""
        if not os.path.exists(sarimax_path) or not os.path.exists(arima_path):
            raise FileNotFoundError("One or both model files not found")
            
        # Load SARIMAX model
        with open(sarimax_path, 'rb') as f:
            sarimax_data = pickle.load(f)
            self.fitted_sarimax = sarimax_data['fitted_model']
            
        # Load ARIMA model
        with open(arima_path, 'rb') as f:
            arima_data = pickle.load(f)
            self.fitted_arima = arima_data['fitted_model']
            
        self.target_column = sarimax_data['target_column']
        print("Both models loaded successfully")

# Example usage:
if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('./dataset/clean_data.csv')
    
    # Convert Convert_time to datetime
    df['Convert_time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    
    # Get user input for the date range
    start_date_str = input("Enter the start date and time (YYYY-MM-DD HH:MM): ")
    end_date_str = input("Enter the end date and time (YYYY-MM-DD HH:MM): ")
    
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        if end_date < start_date:
            print("End date must be after start date")
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
    
    print(f"\nTraining models with data from {start_date_str} to {end_date_str}")
    print(f"Number of hourly data points: {len(sample_data)}")
    
    # Initialize and train the models
    forecaster = TimeSeriesForecaster(sample_data, 'value')
    forecaster.preprocess_data()
    forecaster.train_models()
    
    # Save the trained models
    forecaster.save_models()
    
    # Get user input for prediction period
    try:
        prediction_hours = int(input("\nEnter the number of hours to predict: "))
        if prediction_hours <= 0:
            print("Prediction period must be greater than 0")
            exit()
    except ValueError:
        print("Please enter a valid number of hours")
        exit()
    
    # Make predictions for the specified number of hours
    predictions = forecaster.predict_next_hour(steps=prediction_hours)
    print(f"\nPredictions for the next {prediction_hours} hours after the end date:")
    
    for i in range(prediction_hours):
        next_hour = end_date + pd.Timedelta(hours=i+1)
        print(f"\n{next_hour.strftime('%Y-%m-%d %H:00')}:")
        for model_name, pred in predictions.items():
            print(f"{model_name}: {pred[i]:.2f}")
    
    # Plot results
    forecaster.plot_forecast(forecast_steps=prediction_hours)
    
    # Create a test set for evaluation (last 24 hours of data)
    test_data = sample_data.iloc[-24:].copy()
    train_data = sample_data.iloc[:-24].copy()
    
    # Evaluate the models
    forecaster.evaluate_models(test_data)
