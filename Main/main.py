import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dateutil.parser import parse as parse_datetime
import os

class ZoneForecastSystem:
    def __init__(self, kmeans_path, scaler_path, cluster_csv_path, arima_model_path, zone_cluster_map_path):
        # Load models
        kmeans_bundle = joblib.load(kmeans_path)
        scaler_bundle = joblib.load(scaler_path)
        self.kmeans = kmeans_bundle['kmeans']
        self.scaler = scaler_bundle['scaler']
        self.feature_names = kmeans_bundle['features']

        self.cluster_summary = pd.read_csv(cluster_csv_path)[['cluster', 'performance_label']].drop_duplicates().set_index('cluster')

        with open(arima_model_path, 'rb') as f:
            arima_data = joblib.load(arima_model_path)
            self.fitted_arima = arima_data['fitted_model']
            self.last_train_time = pd.to_datetime(arima_data['last_train_timestamp'])
            self.features = arima_data.get('features', ['hour', 'day_of_week'])

        # Load static zone lookup map
        self.zone_map = pd.read_csv(zone_cluster_map_path)

    def label_zone(self, cluster_id):   
        return self.cluster_summary.loc[cluster_id, 'performance_label']

    def find_performance_label(self, lat, lon):
        df = self.zone_map.copy()
        df['distance'] = ((df['latitude'] - lat) ** 2 + (df['longitude'] - lon) ** 2) ** 0.5
        nearest = df.loc[df['distance'].idxmin()]
        return nearest['performance_label'], nearest['latitude'], nearest['longitude']

    def forecast_throughput(self, start_time, end_time):
        if end_time <= start_time:
            raise ValueError("End time must be after start time.")
        if start_time < self.last_train_time:
            raise ValueError(f"Start time must be after training time: {self.last_train_time}")

        future_index = pd.date_range(start=start_time, end=end_time, freq='h')
        exog = pd.DataFrame(index=future_index)
        exog['hour'] = exog.index.hour
        exog['day_of_week'] = exog.index.dayofweek
        exog['minute'] = exog.index.minute
        exog['hour_sin'] = np.sin(2 * np.pi * exog['hour'] / 24)
        exog['hour_cos'] = np.cos(2 * np.pi * exog['hour'] / 24)
        exog['day_sin'] = np.sin(2 * np.pi * exog['day_of_week'] / 7)
        exog['day_cos'] = np.cos(2 * np.pi * exog['day_of_week'] / 7)
        exog['minute_sin'] = np.sin(2 * np.pi * exog['minute'] / 60)
        exog['minute_cos'] = np.cos(2 * np.pi * exog['minute'] / 60)

        exog = exog[self.features]
        forecast = self.fitted_arima.forecast(steps=len(future_index), exog=exog)
        return forecast, future_index

    def predict_from_csv(self, file_path):
        df = pd.read_csv(file_path)
        required = ['latitude', 'longitude', 'upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec',
                    'upload_transfer_size_mbytes', 'download_transfer_size_rx_mbytes', 'svr1', 'svr2', 'svr3', 'svr4']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing: {col}")

        df['average_latency'] = df[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)
        df['total_throughput'] = df['upload_transfer_size_mbytes'] + df['download_transfer_size_rx_mbytes']
        df['total_bandwidth'] = df['upload_bitrate_mbits/sec'] + df['download_bitrate_rx_mbits/sec']
        features = ['latitude', 'longitude', 'average_latency', 'total_throughput', 'total_bandwidth']
        scaled = self.scaler.transform(df[features])
        clusters = self.kmeans.predict(scaled)
        df['cluster'] = clusters
        df['performance_label'] = [self.label_zone(c) for c in clusters]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {'Low Performance': 'red', 'Moderate': 'orange', 'High Performance': 'green'}
        for label, color in colors.items():
            subset = df[df['performance_label'] == label]
            ax.scatter(subset['longitude'], subset['latitude'], label=label, c=color, s=10)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.set_title("Zone Performance Heatmap")
        plt.tight_layout()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "performance_map.png")
        fig.savefig(output_path)

        window = Toplevel()
        window.title("Zone Heatmap")
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        return output_path

class ZoneGUI:
    def __init__(self, root, system):
        self.root = root
        self.system = system
        root.title("5G Zone Prediction System")


        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)

        self.setup_zone_tab(notebook)
        self.setup_forecast_tab(notebook)
        self.setup_csv_tab(notebook)

    def setup_zone_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Zone Lookup")

        self.entries = {}
        for i, label in enumerate(['Latitude', 'Longitude']):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w')
            e = ttk.Entry(frame)
            e.grid(row=i, column=1)
            self.entries[label.lower()] = e

        self.result_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.result_var).grid(row=3, column=0, columnspan=2)
        ttk.Button(frame, text="Find Zone", command=self.predict_zone).grid(row=2, column=0, columnspan=2)

    def setup_forecast_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Forecast Throughput")

        ttk.Label(frame, text="Start (YYYY-MM-DD HH:MM)").grid(row=0, column=0)
        ttk.Label(frame, text="End (YYYY-MM-DD HH:MM)").grid(row=1, column=0)
        self.start_entry = ttk.Entry(frame)
        self.end_entry = ttk.Entry(frame)
        self.start_entry.grid(row=0, column=1)
        self.end_entry.grid(row=1, column=1)

        self.forecast_text = tk.Text(frame, height=20, width=110)
        self.forecast_text.grid(row=3, column=0, columnspan=2)

        ttk.Button(frame, text="Forecast", command=self.forecast).grid(row=2, column=0, columnspan=2)

    def setup_csv_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="CSV Zone Heatmap")

        ttk.Button(frame, text="Choose CSV & Generate", command=self.load_csv).pack(pady=20)
        self.csv_status = tk.StringVar()
        ttk.Label(frame, textvariable=self.csv_status).pack()

        quit_button = ttk.Button(root, text="Quit", command=root.quit)
        quit_button.pack(pady=10)

    def predict_zone(self):
        try:
            lat_str = self.entries['latitude'].get().strip()
            lon_str = self.entries['longitude'].get().strip()

            if not lat_str or not lon_str:
                raise ValueError("Both latitude and longitude must be provided.")

            lat = float(lat_str)
            lon = float(lon_str)

            # Validate against bounds from zone_cluster_map
            lat_min = self.system.zone_map['latitude'].min()
            lat_max = self.system.zone_map['latitude'].max()
            lon_min = self.system.zone_map['longitude'].min()
            lon_max = self.system.zone_map['longitude'].max()

            if not (lat_min <= lat <= lat_max):
                raise ValueError(f"Latitude must be between {lat_min:.4f} and {lat_max:.4f} (based on available data).")
            if not (lon_min <= lon <= lon_max):
                raise ValueError(f"Longitude must be between {lon_min:.4f} and {lon_max:.4f} (based on available data).")

            label, match_lat, match_lon = self.system.find_performance_label(lat, lon)
            self.result_var.set(f"Closest Match: ({match_lat:.5f}, {match_lon:.5f}) → {label}")

        except ValueError as ve:
            messagebox.showerror("Invalid Input", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def forecast(self):
        try:
            start = parse_datetime(self.start_entry.get())
            end = parse_datetime(self.end_entry.get())
            forecast, times = self.system.forecast_throughput(start, end)
            self.forecast_text.delete(1.0, tk.END)
            for i in range(len(times)):
                self.forecast_text.insert(tk.END, f"{times[i]}: {forecast.iloc[i]:.2f} Mbps\n")
        except Exception as e:
            messagebox.showerror("Forecast Error", str(e))

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                map_path = self.system.predict_from_csv(file_path)
                self.csv_status.set(f"Heatmap saved as: {map_path}")
            except Exception as e:
                self.csv_status.set(f"Error: {e}")

if __name__ == '__main__':
    root = tk.Tk()

    system = ZoneForecastSystem(
        "./TrainedModel/Clustering/cluster_label_kmeans.pkl",
        "./TrainedModel/Clustering/cluster_label_scaler.pkl",
        "./TrainedModel/Clustering/clustering_output.csv",
        "./TrainedModel/TimeSeries/arima_model.pkl",
        "./TrainedModel/Clustering/zone_cluster_map.csv"
    )

    app = ZoneGUI(root, system)
    root.mainloop()
