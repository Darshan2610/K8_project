import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


df = pd.read_csv("kubernetes_performance_metrics_dataset.csv")

features = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "disk_io",
    "network_latency",
    "node_cpu_usage",
    "node_memory_usage",
    "node_temperature"
]
df = df[features].dropna()
df = df.head(500)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(y.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, batch_size=32, verbose=0)

y_pred = model.predict(X)


errors = np.mean(np.abs(y - y_pred), axis=1)
threshold = np.mean(errors) + 2 * np.std(errors)
anomalies = errors > threshold

plt.figure(figsize=(12, 6))
plt.plot(y[:, 0], label="Actual CPU", linewidth=1)
plt.plot(y_pred[:, 0], label="Predicted CPU", linestyle='--')
plt.axhline(y=threshold, color='gray', linestyle=':', label='Anomaly Threshold')
plt.scatter(np.where(anomalies)[0], y[anomalies, 0], color='red', label='Anomaly', zorder=5)
plt.title("Multivariate LSTM CPU Prediction with Anomaly Detection")
plt.xlabel("Time Step")
plt.ylabel("CPU Allocation Efficiency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()