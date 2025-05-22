import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv(r"C:\Users\olive\OneDrive - Swinburne University\AI PROJECT\Week 10\clean_data_clst.csv")

# Select only required columns
selected_columns = ['latitude', 'longitude', 'square_id','average_latency', 'total_bandwidth','total_throughput']
df = df[selected_columns]

# Remove rows where latitude & longitude is 99 999, just in case something slipped through
df = df[~df['latitude'].isin([99, 999]) & ~df['longitude'].isin([99, 999])]

# Normalize relevant numeric columns
columns_to_normalize = ['latitude', 'longitude', 'average_latency', 'total_bandwidth','total_throughput']
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("Data_Clustering_2.csv", index=False)

print("Processed, normalized, and shuffled data saved as Processed_Data.csv")

