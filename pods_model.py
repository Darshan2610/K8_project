import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models


"""
pods_model.py
--------------

Train a lightweight Transformer-based time-series model focused on pod / node
CPU and memory usage, and save:

- pods_scaler.pkl         : MinMaxScaler fitted on the 2-D features
- pods_k8s_model.keras   : Keras model for multi-step prediction

Assumptions
-----------
- The CSV `kubernetes_performance_metrics_dataset.csv` is available in the
  current working directory.
- We only use two features:
    * node_cpu_usage
    * node_memory_usage
- Each row is an equally-spaced time step (e.g. 30s or 1 minute).
- We predict H steps into the future (HORIZON_STEPS).
"""


FILE_PATH = "kubernetes_performance_metrics_dataset.csv"
SEQ_LEN = 20  # history window length
HORIZON_STEPS = 5  # how many future steps we predict (e.g. ~5 minutes)


def load_data():
    df = pd.read_csv(FILE_PATH)
    features = [
        "node_cpu_usage",
        "node_memory_usage",
    ]
    df = df[features].dropna().head(1000)  # keep a small subset for speed

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    joblib.dump(scaler, "pods_scaler.pkl")
    return scaled


def create_sequences_multi_step(data, seq_len=20, horizon=5):
    """
    Build (X, y) pairs where:
      - X[i]  = data[i-seq_len : i]               -> (seq_len, n_features)
      - y[i]  = data[i : i+horizon] (flattened)   -> (horizon * n_features,)
    """
    X, y = [], []
    n = len(data)
    n_feat = data.shape[1]

    for end in range(seq_len, n - horizon):
        start = end - seq_len
        X.append(data[start:end])  # (seq_len, n_feat)
        future = data[end : end + horizon]  # (horizon, n_feat)
        y.append(future.reshape(-1))  # flatten

    return np.array(X), np.array(y)


def build_transformer_model(seq_len, n_features, horizon_steps):
    """
    Simple Transformer encoder block that maps (seq_len, n_features)
    -> (horizon_steps * n_features) regression output.
    """
    inputs = layers.Input(shape=(seq_len, n_features))

    attn = layers.MultiHeadAttention(num_heads=2, key_dim=n_features)(inputs, inputs)
    x = layers.Add()([inputs, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(64, activation="relu")(x)
    ffn = layers.Dense(n_features)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    pooled = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(horizon_steps * n_features)(pooled)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_save():
    scaled = load_data()
    X, y = create_sequences_multi_step(scaled, seq_len=SEQ_LEN, horizon=HORIZON_STEPS)

    if len(X) == 0:
        raise RuntimeError("Not enough data to build sequences; check CSV length.")

    model = build_transformer_model(SEQ_LEN, X.shape[2], HORIZON_STEPS)
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    model.save("pods_k8s_model.keras")
    print("Saved pods_k8s_model.keras and pods_scaler.pkl")


if __name__ == "__main__":
    train_and_save()
