import pandas as pd
import numpy as np
from os import listdir

DATASET_PATH = './model/dataset/'
csvs = [DATASET_PATH + f for f in listdir(DATASET_PATH) if f.endswith('.csv')]

# Combining all csv files into a single dataframe
raw_data = pd.concat([pd.read_csv(csv, low_memory=False) for csv in csvs])

# Features selected
features = ["Day", "Year", "Month", "Date", "hour", "min", "sec",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 
            'Transfer size', 'Transfer size-RX', 'Bitrate', 'Bitrate-RX', "send_data", 'square_id']

# Dropping rows with NaN
raw_data = raw_data[features].dropna()

raw_data['YEAR'] = raw_data['Year'].astype(int)
raw_data['MONTH'] = raw_data['Month'].astype(int)
raw_data['DATE'] = raw_data['Date'].astype(int)
raw_data['HOUR'] = raw_data['hour'].astype(int)
raw_data['MIN'] = raw_data['min'].astype(int)
raw_data['SEC'] = raw_data['sec'].astype(int)

raw_data = raw_data.drop(columns=['Year', 'Month', 'Date', 'hour', 'min', 'sec'])

# two composite features
raw_data['total_throughput'] = raw_data['Bitrate'] + raw_data['Bitrate-RX']
raw_data['total_bandwidth'] = raw_data['Transfer size'] + raw_data['Transfer size-RX']
raw_data['average_latency'] = raw_data[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)

raw_data['DATES'] = pd.to_datetime(raw_data['YEAR'].astype(str) + '-' + raw_data['MONTH'].astype(str) + '-' + raw_data['DATE'].astype(str), format='%Y-%m-%d').dt.date
raw_data['TIME'] = pd.to_datetime(raw_data['HOUR'].astype(str) + ':' + raw_data['MIN'].astype(str) + ':' + raw_data['SEC'].astype(str), format='%H:%M:%S').dt.time
raw_data['Convert_time'] = pd.to_datetime(raw_data['DATES'].astype(str) + ' ' + raw_data['TIME'].astype(str)).dt.strftime('%Y-%m-%d %H:%M:%S')

clean_features = ["Convert_time", "DATES", "TIME", "Day", "YEAR", "MONTH", "DATE", "HOUR", "MIN", "SEC",
            'latitude', 'longitude', 'speed', 'svr1', 'svr2', 'svr3', 'svr4', 'Transfer size', 'Transfer size-RX', 
            'Bitrate', 'Bitrate-RX', "send_data", 'square_id', 'total_throughput', 'total_bandwidth', 'average_latency']

raw_data = raw_data[clean_features]
raw_data = raw_data.rename(columns={
    'Transfer size': 'upload_transfer_size_mbytes',
    'Transfer size-RX': 'download_transfer_size_rx_mbytes',
    'Bitrate': 'upload_bitrate_mbits/sec',
    'Bitrate-RX': 'download_bitrate_rx_mbits/sec',
    'send_data': 'application_data',
    'Day': 'DAY'
})
# print(raw_data.dtypes)
raw_data.to_csv('./model/clean_data_clst.csv', index=False)

# data = pd.read_csv('clean_data_clst.csv')
# data.columns
# data.describe()