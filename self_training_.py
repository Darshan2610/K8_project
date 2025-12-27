# self_training.py
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

SEQ_LEN = 5
HORIZON = 5
CSV = "prom_history.csv"



FEATURES = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "network_latency",
    "node_temperature",
    "node_cpu_usage",
    "node_memory_usage",
]




def create_sequences(data):
    X, y = [], []
    n = len(data)
    for i in range(SEQ_LEN, n - HORIZON):
        X.append(data[i - SEQ_LEN : i])
        y.append(data[i : i + HORIZON].reshape(-1))
    return np.array(X), np.array(y)


def build_model():
    n_feat = len(FEATURES)
    inputs = layers.Input(shape=(SEQ_LEN, n_feat))
    x = layers.MultiHeadAttention(num_heads=4, key_dim=n_feat)(inputs, inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(HORIZON * n_feat)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def retrain():
    df = pd.read_csv(CSV)  # read CSV with headers
# Select only the 7 numeric features for training
    df_features = df[FEATURES]

# Keep timestamps separately if needed
    timestamps = pd.to_datetime(df["timestamp"])

    df = df[FEATURES].dropna()
    if len(df) < 20:
        print("Not enough data to train yet…")
        return

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled)
    if len(X) == 0:
        print("Not enough sequence data.")
        return

    model = build_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    model.save("models/multi_k8s_model.keras")
    joblib.dump(scaler, "models/multi_scaler.pkl")

    print("Retraining complete — model updated.")


if __name__ == "__main__":
    print("Self-training engine started (hourly)…")
    while True:
        retrain()
        time.sleep(120)
