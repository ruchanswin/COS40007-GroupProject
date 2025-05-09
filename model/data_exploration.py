import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)

df = pd.read_csv('./model/dataset/clean_data.csv')
print(df.dtypes)
# df.head()

avg_latency = np.mean(df[['svr1', 'svr2', 'svr3', 'svr4']], axis=1)
# Create a Scatter plot of the latitude and longitude points, coloured by the average latency
im = plt.scatter(x=df['latitude'],
                 y=df['longitude'],
                 s=0.25,
                 vmin=np.min(avg_latency),
                 vmax=np.max(avg_latency),
                 c=avg_latency,
                 alpha=0.5,
                 cmap=cm.plasma)
plt.colorbar(im)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Average Latency (sec)')
plt.axis([-37.800, -37.700, 144.750, 144.850])
plt.savefig('./model/results/lat_lon.png', dpi=300, bbox_inches='tight')
# plt.show()

# Create mutliple Scatter plots for latency, upload and download rates
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

features = [
    ('Average Latency (sec)', avg_latency),
    ('Upload Bitrate (mbits/sec)', df['upload_bitrate_mbits/sec']),
    ('Download Bitrate (mbits/sec)', df['download_bitrate_rx_mbits/sec'])
]

for index, feature in enumerate(features):
    title, column = feature
    axis = axs[index]

    axis.scatter(x=df['latitude'],
                 y=df['longitude'],
                 s=0.25,
                 vmin=np.min(column),
                 vmax=np.max(column),
                 c=column,
                 alpha=0.5,
                 cmap=cm.plasma)
    axis.set_title(title)
    axis.set_xlim(-37.775, -37.725)
    axis.set_ylim(144.775, 144.825)
plt.savefig('./model/results/scatter_plots.png', dpi=300, bbox_inches='tight')
# plt.show()

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

    sns.histplot(column,
                 color=color,
                 kde=True,
                 stat='density',
                 bins=20,
                 ax=axis)
    axis.set_facecolor('w')
    axis.axvline(column.mean(),
                 linestyle='dashed',
                 label='mean',
                 color='k')
    axis.legend(loc='best')
    axis.set_title(title)
    axis.set_xlabel('')
plt.savefig('./model/results/histogram.png', dpi=300, bbox_inches='tight')  
# plt.show()