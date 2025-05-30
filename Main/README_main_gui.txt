
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
│   ├── clustered_output.csv
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
