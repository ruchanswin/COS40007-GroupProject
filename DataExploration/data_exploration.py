import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

# -----------------------------
# Load the data from CSV
df = pd.read_csv('./ProcessedData/clean_data.csv')

# -----------------------------
print("\n--- Data Info ---")
print(df.info())

# -----------------------------
print("\n--- Descriptive Stats ---")
print(df.describe())

# -----------------------------
# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Between Network Performance Features")
plt.savefig('./Results/correlation_heatmap.png', bbox_inches='tight')
# plt.show()

# -----------------------------
# Line plots for each server's latency
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axs[0, 0].plot(df['svr1'], color='blue')
axs[0, 0].set_title('Latency: svr1')
axs[0, 0].set_ylabel('Latency (ms)')
axs[0, 0].grid(True)

axs[0, 1].plot(df['svr2'], color='orange')
axs[0, 1].set_title('Latency: svr2')
axs[0, 1].set_ylabel('Latency (ms)')
axs[0, 1].grid(True)

axs[1, 0].plot(df['svr3'], color='green')
axs[1, 0].set_title('Latency: svr3')
axs[1, 0].set_xlabel('Data Point Index')
axs[1, 0].set_ylabel('Latency (ms)')
axs[1, 0].grid(True)

axs[1, 1].plot(df['svr4'], color='red')
axs[1, 1].set_title('Latency: svr4')
axs[1, 1].set_xlabel('Data Point Index')
axs[1, 1].set_ylabel('Latency (ms)')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('./Results/latency_comparison.png', bbox_inches='tight')
# plt.show()

# -----------------------------
# Scatter plot: Average Latency vs Total Throughput
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='average_latency', y='total_throughput')
plt.title("Average Latency vs Total Throughput")
plt.xlabel("Average Latency (ms)")
plt.ylabel("Total Throughput (Mbps)")
plt.grid(True)
plt.savefig('./Results/latency_vs_throughput.png', bbox_inches='tight')
# plt.show()

# -----------------------------
# Box Plot: Upload vs Download Bitrate
plt.figure(figsize=(6, 4))
df[['upload_bitrate_mbits/sec', 'download_bitrate_rx_mbits/sec']].plot(kind='box')
plt.title('Upload vs Download Bitrate Distribution')
plt.ylabel("Mbps")
plt.grid(True)
plt.savefig('./Results/bitrate_distribution.png', bbox_inches='tight')
# plt.show()

# -----------------------------
# Scatter plots of Latency by Server
lat_limits = [-37.800, -37.700]
lon_limits = [144.750, 144.850]
servers = ['svr1', 'svr2', 'svr3', 'svr4']

for server in servers:
    plt.figure(figsize=(8, 6))
    im = plt.scatter(
        x=df['latitude'],
        y=df['longitude'],
        s=0.25,
        c=df[server],
        cmap=cm.plasma,
        alpha=0.5,
        vmin=df[server].min(),
        vmax=df[server].max()
    )
    plt.colorbar(im, label='Latency (ms)')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Latency Scatter Plot for {server}')
    plt.axis([lat_limits[0], lat_limits[1], lon_limits[0], lon_limits[1]])
    plt.savefig(f'./Results/{server}_lat_lon.png', dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Scatter Plot: Average Latency
avg_latency = np.mean(df[['svr1', 'svr2', 'svr3', 'svr4']], axis=1)
im = plt.scatter(
    x=df['latitude'],
    y=df['longitude'],
    s=0.25,
    vmin=np.min(avg_latency),
    vmax=np.max(avg_latency),
    c=avg_latency,
    alpha=0.5,
    cmap=cm.plasma
)
plt.colorbar(im)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Average Latency (sec)')
plt.axis([-37.800, -37.700, 144.750, 144.850])
plt.savefig('./Results/lat_lon.png', dpi=300, bbox_inches='tight')
# plt.show()

# -----------------------------
# Scatter Plots: Latency, Upload, and Download
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
features = [
    ('Average Latency (sec)', avg_latency),
    ('Upload Bitrate (mbits/sec)', df['upload_bitrate_mbits/sec']),
    ('Download Bitrate (mbits/sec)', df['download_bitrate_rx_mbits/sec'])
]

for index, feature in enumerate(features):
    title, column = feature
    axis = axs[index]

    axis.scatter(
        x=df['latitude'],
        y=df['longitude'],
        s=0.25,
        vmin=np.min(column),
        vmax=np.max(column),
        c=column,
        alpha=0.5,
        cmap=cm.plasma
    )
    axis.set_title(title)
    axis.set_xlim(-37.775, -37.725)
    axis.set_ylim(144.775, 144.825)

plt.savefig('./Results/scatter_plots.png', dpi=300, bbox_inches='tight')
# plt.show()

# -----------------------------
# Histograms with KDE for Transfer Sizes and Bitrates
features = [
    ('Upload Transfer Size (mbytes)', 'b', df['upload_transfer_size_mbytes']),
    ('Upload Bitrate (mbits/sec)', 'r', df['upload_bitrate_mbits/sec']),
    ('Download Transfer Size (mbytes)', 'g', df['download_transfer_size_rx_mbytes']),
    ('Download Bitrate (mbits/sec)', 'c', df['download_bitrate_rx_mbits/sec'])
]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.ravel()

for axis, feature in zip(axs, features):
    title, color, column = feature

    # Remove outliers
    column = column[column > 0]
    column = column[np.abs(stats.zscore(column)) < 3]

    sns.histplot(
        column,
        color=color,
        kde=True,
        stat='density',
        bins=20,
        ax=axis
    )
    axis.set_facecolor('w')
    axis.axvline(column.mean(), linestyle='dashed', label='mean', color='k')
    axis.legend(loc='best')
    axis.set_title(title)
    axis.set_xlabel('')

plt.savefig('./Results/histogram.png', dpi=300, bbox_inches='tight')
# plt.show()