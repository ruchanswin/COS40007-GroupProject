"""
This script cleans the raw data to use for training the model.
"""

import pandas as pd
from os import listdir, path

# Path to the directory containing the raw CSV files directly (no zip)
RAW_DATA_PATH = './dataset/raw_data/'

# Get list of all CSV files in the raw data folder
csvs = [path.join(RAW_DATA_PATH, f) for f in listdir(RAW_DATA_PATH) if f.endswith('.csv')]

# Combine all CSV files into a single dataframe
raw_data = pd.concat([pd.read_csv(csv, low_memory=False) for csv in csvs])

# Features selected
features = ["Day", "Year", "Month", "Date", "hour", "min", "sec",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
            'Transfer size', 'Transfer size-RX', 'Bitrate', 'Bitrate-RX', "send_data", 'square_id']

# Dropping rows with NaN
raw_data = raw_data[features].dropna()

# Remove rows with invalid values
raw_data = raw_data[(raw_data['latitude'] != 99.0) & (raw_data['longitude'] != 999.0)]
raw_data = raw_data[raw_data['speed'] != -1]
raw_data = raw_data[(raw_data['svr1'] != 1000) &
                    (raw_data['svr2'] != 1000) &
                    (raw_data['svr3'] != 1000) &
                    (raw_data['svr4'] != 1000)]

# Convert columns to int
raw_data['YEAR'] = raw_data['Year'].astype(int)
raw_data['MONTH'] = raw_data['Month'].astype(int)
raw_data['DATE'] = raw_data['Date'].astype(int)
raw_data['HOUR'] = raw_data['hour'].astype(int)
raw_data['MIN'] = raw_data['min'].astype(int)
raw_data['SEC'] = raw_data['sec'].astype(int)

raw_data = raw_data.drop(columns=['Year', 'Month', 'Date', 'hour', 'min', 'sec'])

# Create new features
raw_data['total_throughput'] = raw_data['Bitrate'] + raw_data['Bitrate-RX']
raw_data['total_data_transferred'] = raw_data['Transfer size'] + raw_data['Transfer size-RX']
raw_data['average_latency'] = raw_data[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)

raw_data['DATE'] = pd.to_datetime(raw_data['YEAR'].astype(str) + '-' + raw_data['MONTH'].astype(str) + '-' + raw_data['DATE'].astype(str), format='%Y-%m-%d').dt.date
raw_data['TIME'] = pd.to_datetime(raw_data['HOUR'].astype(str) + ':' + raw_data['MIN'].astype(str) + ':' + raw_data['SEC'].astype(str), format='%H:%M:%S').dt.time

# Selecting clean features
clean_features = ['DATE', 'TIME', 'Day', 'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
                  'average_latency', 'Transfer size', 'Transfer size-RX', 'total_data_transferred',
                  'Bitrate', 'Bitrate-RX', 'total_throughput', 'send_data', 'square_id']

raw_data = raw_data[clean_features]

# Renaming columns for clarity
raw_data = raw_data.rename(columns={
    'Transfer size': 'upload_transfer_size_mbytes',
    'Transfer size-RX': 'download_transfer_size_rx_mbytes',
    'Bitrate': 'upload_bitrate_mbits/sec',
    'Bitrate-RX': 'download_bitrate_rx_mbits/sec',
    'send_data': 'application_data',
    'Day': 'DAY'
})
# Save the cleaned data to a new CSV file
clean_csv_path = './dataset/clean_data.csv'
raw_data.to_csv(clean_csv_path, index=False)

print(f"Cleaned data saved to {clean_csv_path}")