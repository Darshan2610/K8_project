import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# --- Config ---
FILE_PATH = "prom_history.csv"
SEQ_LEN = 20
HORIZON_STEPS = 5
SCALER_PATH = "multi_scaler.pkl"
MODEL_PATH = "multi_k8s_model.keras"
MAX_ROWS = 2000
EPOCHS = 30
BATCH_SIZE = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Use the exact 7 legacy names already present in your CSV
FEATURES = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "disk_io",
    "network_latency",
    "node_cpu_usage",
    "node_memory_usage",
    "node_temperature",
]


def load_and_prepare_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError("CSV missing columns: " + ", ".join(missing))

    # Keep only feature columns, drop NaNs
    df = df[FEATURES].dropna().copy()

    # Use last MAX_ROWS
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS).reset_index(drop=True)

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Clip extremes to stabilize scaling
    df["cpu_allocation_efficiency"] = df["cpu_allocation_efficiency"].clip(0.0, 10.0)
    df["memory_allocation_efficiency"] = df["memory_allocation_efficiency"].clip(
        0.0, 10.0
    )
    df["disk_io"] = df["disk_io"].clip(0.0, 1e12)
    df["network_latency"] = df["network_latency"].clip(0.0, 1e6)
    df["node_cpu_usage"] = df["node_cpu_usage"].clip(0.0, 100.0)
    df["node_memory_usage"] = df["node_memory_usage"].clip(0.0, 100.0)
    df["node_temperature"] = df["node_temperature"].clip(0.0, 200.0)

    return df


def fit_and_save_scaler(df: pd.DataFrame, scaler_path: str) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df.values)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    return scaler


def create_sequences(data: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    n = data.shape[0]
    for end in range(seq_len, n - horizon + 1):
        start = end - seq_len
        X.append(data[start:end])
        y.append(data[end : end + horizon].reshape(-1))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


def build_model(seq_len: int, n_feat: int, horizon: int):
    inputs = layers.Input(shape=(seq_len, n_feat))
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(1, n_feat // 2))(
        inputs, inputs
    )
    x = layers.Add()([inputs, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(n_feat)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(horizon * n_feat)(pooled)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_save():
    df = load_and_prepare_df(FILE_PATH)
    if df.shape[0] < SEQ_LEN + HORIZON_STEPS:
        raise RuntimeError(
            f"Not enough rows. Need at least {SEQ_LEN + HORIZON_STEPS}, got {df.shape[0]}"
        )

    scaler = fit_and_save_scaler(df, SCALER_PATH)
    scaled = scaler.transform(df.values)

    X, y = create_sequences(scaled, SEQ_LEN, HORIZON_STEPS)
    print(f"Prepared sequences: X.shape={X.shape}, y.shape={y.shape}")

    model = build_model(SEQ_LEN, X.shape[2], HORIZON_STEPS)
    model.summary()
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


if __name__ == "__main__":
    train_and_save()
