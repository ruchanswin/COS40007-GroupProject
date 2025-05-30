import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

# Load raw unscaled data
df = pd.read_csv("./ProcessedData/clean_data.csv")

# Select features
feature_cols = ['latitude', 'longitude', 'average_latency', 'total_throughput', 'total_bandwidth']
X_raw = df[feature_cols].copy()

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Determine Best K with multiple metrics
best_k = 2
best_score = -1
samples_per_cluster = 3000
print("Evaluating clustering metrics for K in range 3 to 10:")

# Store metrics for each k
metrics = {
    'k': [],
    'silhouette': [],
    'calinski_harabasz': [],
    'davies_bouldin': []
}

for k in range(3, 11):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_tmp = kmeans_tmp.fit_predict(X_scaled)
    sampled_indices = []

    for cluster_id in np.unique(labels_tmp):
        cluster_rows = np.where(labels_tmp == cluster_id)[0]
        chosen = np.random.choice(cluster_rows, size=min(samples_per_cluster, len(cluster_rows)), replace=False)
        sampled_indices.extend(chosen)

    X_sample = X_scaled[sampled_indices]
    labels_sample = labels_tmp[sampled_indices]
    
    # Calculate metrics
    sil_score = silhouette_score(X_sample, labels_sample)
    ch_score = calinski_harabasz_score(X_sample, labels_sample)
    db_score = davies_bouldin_score(X_sample, labels_sample)
    
    metrics['k'].append(k)
    metrics['silhouette'].append(sil_score)
    metrics['calinski_harabasz'].append(ch_score)
    metrics['davies_bouldin'].append(db_score)
    
    print(f"\nK = {k}:")
    print(f" - Silhouette Score = {sil_score:.4f}")
    print(f" - Calinski-Harabasz Score = {ch_score:.4f}")
    print(f" - Davies-Bouldin Score = {db_score:.4f}")

    if sil_score > best_score:
        best_k = k
        best_score = sil_score

print(f"\nBest K selected: {best_k} with silhouette score = {best_score:.4f}")

# Train final KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(X_scaled)

# Create Cluster Summary
cluster_summary = df.groupby('cluster')[['average_latency', 'total_throughput']].mean()
cluster_summary['score'] = cluster_summary['total_throughput'] / cluster_summary['average_latency']
ranking_map = cluster_summary['score'].rank(ascending=False).astype(int) - 1
label_map = {0: 'High Performance', 1: 'Moderate', 2: 'Low Performance'}
cluster_summary['quality_rank'] = ranking_map
cluster_summary['performance_label'] = ranking_map.map(label_map)

# Merge performance labels back
df = df.merge(cluster_summary[['performance_label']], left_on='cluster', right_index=True)

# Save files
output_dir = "./TrainedModel/Clustering/"
os.makedirs(output_dir, exist_ok=True)

# Save evaluation metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"{output_dir}clustering_metrics.csv", index=False)

# Save model and data
df.to_csv(f"{output_dir}clustering_output.csv", index=False)
joblib.dump({'scaler': scaler, 'features': feature_cols}, f"{output_dir}cluster_label_scaler.pkl")
joblib.dump({'kmeans': kmeans, 'features': feature_cols}, f"{output_dir}cluster_label_kmeans.pkl")

zone_map = df.drop_duplicates(subset=['latitude', 'longitude'])[['latitude', 'longitude', 'cluster', 'performance_label']]
zone_map.to_csv(f"{output_dir}zone_cluster_map.csv", index=False)

print("\nTraining complete. Files saved.")
