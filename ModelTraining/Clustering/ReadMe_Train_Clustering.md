
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

1. Load the cleaned dataset (`clean_data_training.csv`)
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

- Ensure input file: `clean_data_training.csv` is placed correctly
- The model is later used by the GUI to label new inputs based on location and network metrics
