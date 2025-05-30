
# data_cleanup.py

This script processes raw 5G network performance data files into a clean and structured dataset ready for clustering and time-series forecasting tasks.

---

## Purpose

To consolidate and clean raw network measurement data from multiple CSV files—merging them, extracting essential features, computing derived metrics, and saving a cleaned dataset for use in machine learning pipelines.

---

## Input

- **Source**: All CSV files must be located in the following OneDrive folder:
  > [OneDrive Theme 5 Raw Data Folder](https://liveswinburneeduau-my.sharepoint.com/shared?id=%2Fpersonal%2Ffforkan%5Fswin%5Fedu%5Fau%2FDocuments%2FCOS40007%2FDesign%20Project%2FTheme5%2Fdata&listurl=%2Fpersonal%2Ffforkan%5Fswin%5Fedu%5Fau%2FDocuments&login_hint=104106821%40student%2Eswin%2Eedu%2Eau&source=waffle)

- **Important**:
  - Before running the script, download the raw `.csv` files from the OneDrive folder.
  - Place them in the local path:  
    ```plaintext
    <Local Directory>/5G Zone Prediction System/RawData
    ```

---

## Updated Directory Configuration

```plaintext
5G Zone Prediction System/
├── DataExploration/
│   ├── data_cleanup.py
├── RawData/
│   └── data.csv<list>                <-- Raw CSV files from OneDrive go here
├── ProcessedData/
│   └── clean_data.csv  <-- Cleaned output will be saved here
```

---

## Output

- File: `5G Zone Prediction System/ProcessedData/clean_data.csv`
- Contains cleaned and feature-enriched records:
  - Time features, GPS coordinates, speed, server latencies, transfer sizes, bitrate metrics
  - Derived metrics: `total_throughput`, `total_bandwidth`, `average_latency`

---

## How to Run

1. Ensure you have downloaded and placed the raw data in `./5G Zone Prediction System/RawData`
2. Execute:
   ```bash
   python data_cleanup.py
   ```

---

## Notes

- This script assumes all CSVs share the same schema and have no conflicting header definitions.
- `Convert_time` is the key datetime field for temporal analysis and forecasting.
- The cleaned dataset is used by:
  - Kmean training pipeline

---


# Train-Clustering.py

This Jupyter Notebook trains a KMeans clustering model to classify geographical zones based on 5G network performance metrics.

---

## Location

`<Local Directory>/5G Zone Prediction System/ModelTraining/Clustering/Train-Clustering.py`

---

## Purpose

To group data points representing physical locations into performance zones using unsupervised KMeans clustering.

---

## Features Used

- `latitude`, `longitude`
- `average_latency`
- `total_throughput`
- `total_bandwidth`

---

## Workflow

1. Load the cleaned dataset (`clean_data.csv`)
2. Apply MinMax scaling to selected features
3. Train multiple KMeans models (K = 1 to 10)
4. Evaluate using silhouette scores
5. Save the best performing model and scaler

---

## Outputs

- Trained KMeans model and scaler using `joblib`
- CSV summary of clusters with performance labels

---

## Usage Notes

- Ensure input file: `clean_data.csv` is placed correctly
- The model is later used by the GUI to label new inputs based on location and network metrics


# Train-TimeSeries.py

This notebook trains an ARIMA model for time-series forecasting of 5G network performance (total throughput) using historical data.

---

## Location

`<Local Directory>/5G Zone Prediction System/ModelTraining/TimeSeries/Train-TimeSeries.py`

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



# main.py - 5G Zone Prediction System GUI

This script is the main entry point for the 5G Zone Prediction System, featuring a GUI for interacting with trained clustering and time series models.

---

## Location

`<Local Directory>/5G Zone Prediction System/main.py`

---

## Purpose

To provide a user-friendly interface to:
- Lookup performance zone of a location using KMeans clustering
- Forecast future throughput using an ARIMA model
- Visualize performance heatmaps from CSV datasets

---

## Models Integrated

- **KMeans Clustering Model** (for zone labeling)
- **ARIMA Time Series Model** (for throughput forecasting)

All models are pre-trained and loaded from the `TrainedModel/` directory.

---

## GUI Tabs Overview

### Zone Lookup
- Enter latitude and longitude
- Find closest matching zone and its performance label

### Forecast Throughput
- Input start and end time (e.g., `2025-06-01 14:00`)
- Output hourly throughput predictions using ARIMA

### CSV Zone Heatmap
- Load CSV with raw performance metrics
- Predict zones for all entries
- Display and save heatmap image (`performance_map.png`)

---

## Required Files

Ensure the following trained assets are available:

```
TrainedModel/
├── Clustering/
│   ├── cluster_label_kmeans.pkl
│   ├── cluster_label_scaler.pkl
│   ├── clustering_output.csv
│   └── zone_cluster_map.csv
└── TimeSeries/
    └── arima_model.pkl
```

---

## How to Run

```bash
python main.py
```

> Python GUI will launch with 3 main functions.

---

## Notes

- Valid latitude/longitude bounds are checked dynamically using `zone_cluster_map.csv`
- The output `performance_map.png` will be saved in the script's directory
- Required CSV for heatmap must include:
  - `latitude`, `longitude`
  - `upload_bitrate_mbits/sec`, `download_bitrate_rx_mbits/sec`
  - `upload_transfer_size_mbytes`, `download_transfer_size_rx_mbytes`
  - `svr1` to `svr4`

---

