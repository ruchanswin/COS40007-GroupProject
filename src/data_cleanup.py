"""
This script cleans the raw data by performing the following steps:
1. Combines all CSV files in the specified directory into a single DataFrame.
2. Selects specific features from the DataFrame.
3. Drops rows with NaN values.
4. Removes rows with invalid values for latitude, longitude, speed, and server metrics.
5. Combined Bitrate and Transfer size columns to create new features.
6. Converts date and time columns to appropriate formats.
7. Renames columns for clarity.
8. Saves the cleaned DataFrame to a new CSV file.
"""

import pandas as pd
from os import listdir

# Path to the directory containing the raw data CSV files
DATASET_PATH = './data/raw_data'
csvs = [DATASET_PATH + f for f in listdir(DATASET_PATH) if f.endswith('.csv')]

# Combining all csv files into a single dataframe
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
clean_features = ['DATE', 'TIME', 'Day','latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
                  'average_latency', 'Transfer size', 'Transfer size-RX', 'total_data_transferred',
                  'Bitrate', 'Bitrate-RX','total_throughput', 'send_data', 'square_id']

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
raw_data.to_csv('./model/clean_data_clst.csv', index=False)
