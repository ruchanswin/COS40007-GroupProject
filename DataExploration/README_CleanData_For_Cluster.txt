
# CleanData_For_Cluster.py

This script processes raw 5G network performance data files into a clean and structured dataset ready for clustering and time-series forecasting tasks.

---

## Purpose

To consolidate and clean raw network measurement data from multiple CSV files—merging them, extracting essential features, computing derived metrics, and saving a cleaned dataset for use in machine learning pipelines.

---

## Input

- 📦 **Source**: All CSV files must be located in the following OneDrive folder:
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
│   ├── CleanData_For_Cluster.py
├── RawData/
│   └── data.csv<list>                <-- Raw CSV files from OneDrive go here
├── ProcessedData/
│   └── clean_data_clst.csv  <-- Cleaned output will be saved here
```

---

## Output

- File: `5G Zone Prediction System/ProcessedData/clean_data_clst.csv`
- Contains cleaned and feature-enriched records:
  - Time features, GPS coordinates, speed, server latencies, transfer sizes, bitrate metrics
  - Derived metrics: `total_throughput`, `total_bandwidth`, `average_latency`

---

## How to Run

1. Ensure you have downloaded and placed the raw data in `./Theme5/data/`
2. Execute:
   ```bash
   python CleanData_For_Cluster.py
   ```

---

## Notes

- This script assumes all CSVs share the same schema and have no conflicting header definitions.
- `Convert_time` is the key datetime field for temporal analysis and forecasting.
- The cleaned dataset is used by:
  - `Data_Clustering_2.csv` generator
  - ARIMA training pipeline

---
