import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense, GRU
from tensorflow.keras import Sequential, Model

pd.set_option('display.max_columns', None)
df = pd.read_csv('./model/dataset/clean_data.csv')
df.head()

df['square_id'].value_counts()
df = df.loc[df['square_id'] == 'square_111669149768']
features = [
    'svr1',
    'svr2',
    'svr3',
    'svr4',
    'upload_bitrate_mbits/sec',
    'download_bitrate_rx_mbits/sec'
]

for feature in features:
    df[feature] = df[feature].replace([0, 1000], np.nan).ffill()

df.dropna(inplace=True)
# df.head()
df['Date Time'] = pd.to_datetime(df['DATES'] + ' ' + df['TIME'])
# df.head()
df = df[[
    'svr1',
    'svr2',
    'svr3',
    'svr4',
    'upload_bitrate_mbits/sec',
    'download_bitrate_rx_mbits/sec',
    'Date Time'
]]
# df.head()
df = df.groupby('Date Time').agg('mean')
# df.head()
df['Average Server Latency'] = df[['svr1', 'svr2', 'svr3', 'svr4']].mean(axis=1)
df['Total Throughput'] = df['upload_bitrate_mbits/sec'] + df['download_bitrate_rx_mbits/sec']
# df.head()
features = [
    'Average Server Latency',
    'Total Throughput'
]

df = df[features]
# df.head()
df = df.resample('1min').mean().dropna()
# df.head()

months_in_year = 12
df['Month (Sin)'] = np.sin(2 * np.pi * df.index.month / months_in_year)
df['Month (Cos)'] = np.cos(2 * np.pi * df.index.month / months_in_year)

days_in_month = 30
df['Day (Sin)'] = np.sin(2 * np.pi * df.index.day / days_in_month)
df['Day (Cos)'] = np.cos(2 * np.pi * df.index.day / days_in_month)

hours_in_day = 24
df['Hour (Sin)'] = np.sin(2 * np.pi * df.index.hour / hours_in_day)
df['Hour (Cos)'] = np.cos(2 * np.pi * df.index.hour / hours_in_day)

minutes_in_hour = 60
df['Minute (Sin)'] = np.sin(2 * np.pi * df.index.minute / minutes_in_hour)
df['Minute (Cos)'] = np.cos(2 * np.pi * df.index.minute / minutes_in_hour)

# df.head()

features = [
    ('Average Server Latency', 'blue'),
    ('Total Throughput', 'orange'),
]

def visualise_data(data):
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(20, 5)
    )

    for i in range(len(features)):
        key, color = features[i]
        t_data = data[key].tail(60 * 24)
        ax = t_data.plot(
            use_index=False,
            ax=axes[i],
            color=color,
            title=key,
            rot=25
        )

        ax.legend([key])
    plt.savefig('./model/results/data_visualisation.png')
    plt.tight_layout()

visualise_data(df)

sampling_rate = 1

train_size = 0.75
train_split = int(train_size * int(df.shape[0]))

past = 60
future = 1
batch_size = 128
epochs = 40

selected_features = [
    'Month (Sin)',
    'Month (Cos)',
    'Day (Sin)',
    'Day (Cos)',
    'Hour (Sin)',
    'Hour (Cos)',
    'Minute (Sin)',
    'Minute (Cos)',
    'Average Server Latency',
    'Total Throughput'
]

features = df[selected_features]
# features.head()

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0:train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(len(selected_features))]].values
y_train = features.iloc[start:end][[9]]

sequence_length = int(past / sampling_rate)

training_dataset = timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_validation = val_data.iloc[:x_end][[i for i in range(len(selected_features))]].values
y_validation = features.iloc[label_start:][[9]]

validation_dataset = timeseries_dataset_from_array(
    x_validation,
    y_validation,
    sequence_length=sequence_length,
    sampling_rate=sampling_rate,
    batch_size=batch_size
)

for batch in validation_dataset.take(1):
    inputs, targets = batch

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)

# Build the LSTM model
inputs = Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = GRU(32)(inputs)
outputs = Dense(1)(lstm_out)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='mse')
model.summary()

path_checkpoint = "./model/pickle/timeseries_model_lstm.weights.h5"
es_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    training_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[es_callback, modelckpt_callback]
)

def visualize_loss(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./model/results/loss_visualization.png')
    # plt.show()

visualize_loss(history, 'Training and Validation Loss')

predictions = []
ground_truth = []

for batch in validation_dataset.take(1):
    inputs, targets = batch
    preds = model.predict(inputs)
    predictions.extend(preds.flatten())
    ground_truth.extend(targets.numpy().flatten())

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

data_duration = 60
prediction_duration = 60

validation_data = ground_truth[-data_duration:]
prediction = predictions[:prediction_duration]
ground_truth = ground_truth[-data_duration:][:prediction_duration]

plt.figure(figsize=(10, 5))

plt.plot(range(-data_duration, 0), validation_data, color='blue', label="Validation Data")
plt.plot(range(0, prediction_duration), prediction, color='orange', label="Predicted Data")
plt.plot(range(0, prediction_duration), ground_truth, color='green', label="Ground Truth")

plt.title('Prediction')
plt.xlabel('Time (Minute)')
plt.ylabel('Total Throughput (mbits/sec)')
plt.legend()
plt.savefig('./model/results/prediction_visualization.png')
# plt.show()