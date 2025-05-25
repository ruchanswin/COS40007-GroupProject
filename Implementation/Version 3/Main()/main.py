import pandas as pd
import numpy as np
import joblib

def load_cluster_summary():
    # Load cluster summary from the clustered output CSV
    df = pd.read_csv("./SaveModel/clustered_50k_output.csv")
    summary = df.groupby('cluster')[['average_latency', 'total_throughput']].mean()
    return summary

def label_zone(cluster_id, cluster_summary):
    throughput = cluster_summary.loc[cluster_id, 'total_throughput']
    latency = cluster_summary.loc[cluster_id, 'average_latency']
    avg_throughput = cluster_summary['total_throughput'].mean()
    avg_latency = cluster_summary['average_latency'].mean()

    if throughput > avg_throughput and latency < avg_latency:
        return 'High Performance'
    elif throughput < avg_throughput and latency > avg_latency:
        return 'Low Performance'
    else:
        return 'Moderate'

def main():
    # Load trained model and scaler
    kmeans = joblib.load("./SaveModel/kmeans_model.pkl")
    scaler = joblib.load("./SaveModel/scaler.pkl")
    cluster_summary = load_cluster_summary()

    while True:
        print("\nWelcome to the 5G Performance Prediction System")
        print("Options:")
        print("1. Predict cluster and zone")
        print("0. Exit")

        choice = input("Enter your choice (0 or 1): ")

        if choice == "1":
            try:
                avg_latency = float(input("Enter average latency (ms): "))
                bandwidth = float(input("Enter total bandwidth: "))
                throughput = float(input("Enter total throughput: "))
                latitude = float(input("Enter latitude: "))
                longitude = float(input("Enter longitude: "))

                # Create input feature array
                input_data = pd.DataFrame([[
                    avg_latency, bandwidth, throughput, latitude, longitude
                ]], columns=['average_latency', 'total_bandwidth', 'total_throughput', 'latitude', 'longitude'])

                # Normalize and predict
                input_scaled = scaler.transform(input_data)
                cluster = kmeans.predict(input_scaled)[0]

                # Get performance label
                zone = label_zone(cluster, cluster_summary)

                print(f"\nPredicted Group: {cluster}")
                print(f"Predicted Zone: {zone}")

            except ValueError:
                print("Invalid input! Please enter numeric values.")

        elif choice == "0":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid option. Please choose 0 or 1.")

if __name__ == "__main__":
    main()
