# import os
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt

# # ---------------- CONFIG ----------------
# CSV_PATH = "prom_history.csv"
# MODEL_PATH = "models/multi_k8s_model.keras"
# SCALER_PATH = "models/multi_scaler.pkl"

# SEQ_LEN = 5
# HORIZON_STEPS = 5
# TEST_SPLIT_RATIO = 0.2

# FEATURES = [
#     "cpu_allocation_efficiency",
#     "memory_allocation_efficiency",
#     "network_latency",
#     "node_cpu_usage",
#     "node_memory_usage",
#     "node_temperature",
# ]

# # Thresholds used by remediation logic
# THRESHOLDS = {
#     "node_cpu_usage": 0.8,
#     "node_memory_usage": 0.8,
# }

# # ----------------------------------------


# def load_data():
#     if not os.path.exists(CSV_PATH):
#         raise FileNotFoundError("prom_history.csv not found")

#     df = pd.read_csv(CSV_PATH)

#     missing = [c for c in FEATURES if c not in df.columns]
#     if missing:
#         raise RuntimeError(f"Missing columns: {missing}")

#     df = df[FEATURES].astype(float).dropna()

#     if len(df) < SEQ_LEN + HORIZON_STEPS:
#         raise RuntimeError("Not enough data for evaluation")

#     return df.values


# def create_sequences(data):
#     X, y = [], []
#     for i in range(len(data) - SEQ_LEN - HORIZON_STEPS + 1):
#         X.append(data[i : i + SEQ_LEN])
#         y.append(data[i + SEQ_LEN : i + SEQ_LEN + HORIZON_STEPS])
#     return np.array(X), np.array(y)


# def threshold_accuracy(actual, predicted, threshold):
#     actual_hit = actual > threshold
#     predicted_hit = predicted > (threshold * 0.9)  # early warning buffer
#     return np.mean(actual_hit == predicted_hit) * 100


# def evaluate():
#     print("Loading data...")
#     raw_data = load_data()

#     print("Loading scaler and model...")
#     scaler = joblib.load(SCALER_PATH)
#     model = tf.keras.models.load_model(MODEL_PATH)

#     scaled = scaler.transform(raw_data)

#     X, y = create_sequences(scaled)

#     split = int(len(X) * (1 - TEST_SPLIT_RATIO))
#     X_test = X[split:]
#     y_test = y[split:]

#     print(f"Evaluating on {len(X_test)} sequences")

#     y_pred = model.predict(X_test, verbose=0)

#     y_test_2d = y_test.reshape(-1, len(FEATURES))
#     y_pred_2d = y_pred.reshape(-1, len(FEATURES))

#     y_test_inv = scaler.inverse_transform(y_test_2d)
#     y_pred_inv = scaler.inverse_transform(y_pred_2d)

#     print("\n===== MODEL EVALUATION =====")

#     plt.figure(figsize=(14, 10))

#     for i, feat in enumerate(FEATURES):
#         actual = y_test_inv[:, i]
#         pred = y_pred_inv[:, i]

#         mae = mean_absolute_error(actual, pred)
#         rmse = np.sqrt(mean_squared_error(actual, pred))

#         print(f"\n{feat}")
#         print(f"  MAE  : {mae:.4f}")
#         print(f"  RMSE : {rmse:.4f}")

#         if feat in THRESHOLDS:
#             acc = threshold_accuracy(actual, pred, THRESHOLDS[feat])
#             print(f"  Threshold Prediction Accuracy : {acc:.2f}%")

#         # Plot
#         plt.subplot(3, 2, i + 1)
#         plt.plot(actual[:200], label="Actual")
#         plt.plot(pred[:200], label="Predicted")
#         plt.title(feat)
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig("model_evaluation_plot.png")
#     plt.close()

#     print("\nEvaluation complete.")
#     print("Saved: model_evaluation_plot.png")


# if __name__ == "__main__":
#     evaluate()


import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ================= CONFIG =================

CSV_PATH = "prom_history.csv"
MODEL_PATH = "models/multi_k8s_model.keras"
SCALER_PATH = "models/multi_scaler.pkl"

SEQ_LEN = 20
HORIZON_STEPS = 5
TEST_SPLIT_RATIO = 0.2

FEATURES = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "network_latency",
    "node_cpu_usage",
    "node_memory_usage",
    "node_temperature",
]

# These thresholds MUST be in REAL (inverse-scaled) units
THRESHOLDS = {
    "node_cpu_usage": 0.7,
    "node_memory_usage": 0.7,
}

PLOT_POINTS = 200

# ==========================================





def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df[FEATURES].astype(float).dropna()

    if len(df) < SEQ_LEN + HORIZON_STEPS:
        raise RuntimeError("Not enough data to evaluate model")

    return df


def create_sequences(data):
    X, y = [], []

    for i in range(len(data) - SEQ_LEN - HORIZON_STEPS + 1):
        X.append(data[i : i + SEQ_LEN])
        y.append(data[i + SEQ_LEN : i + SEQ_LEN + HORIZON_STEPS])

    return np.array(X), np.array(y)


def threshold_accuracy(actual, predicted, threshold):
    """
    Checks whether the model correctly predicts
    stress vs no-stress situations.
    """
    actual_stress = actual >= threshold
    predicted_stress = predicted >= threshold * 0.9  # early-warning buffer

    return np.mean(actual_stress == predicted_stress) * 100


def evaluate():
    print("Loading data...")
    df = load_data()
    raw_values = df.values

    print("Loading model and scaler...")
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    # ---------- SCALE INPUT ----------
    scaled_values = scaler.transform(raw_values)

    # ---------- SEQUENCES ----------
    X, y = create_sequences(scaled_values)

    split_index = int(len(X) * (1 - TEST_SPLIT_RATIO))
    X_test = X[split_index:]
    y_test = y[split_index:]

    print(f"Evaluating on {len(X_test)} sequences")

    # ---------- PREDICTION ----------
    y_pred = model.predict(X_test, verbose=0)

    # ---------- RESHAPE ----------
    y_test_flat = y_test.reshape(-1, len(FEATURES))
    y_pred_flat = y_pred.reshape(-1, len(FEATURES))

    # ---------- INVERSE SCALE ----------
    y_test_real = scaler.inverse_transform(y_test_flat)
    y_pred_real = scaler.inverse_transform(y_pred_flat)

    print("\n========== MODEL EVALUATION ==========")

    plt.figure(figsize=(15, 10))

    for idx, feature in enumerate(FEATURES):
        actual = y_test_real[:, idx]
        predicted = y_pred_real[:, idx]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        print(f"\nFeature: {feature}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")

        if feature in THRESHOLDS:
            acc = threshold_accuracy(actual, predicted, THRESHOLDS[feature])
            print(f"  Stress Prediction Accuracy: {acc:.2f}%")

        # ---------- PLOT ----------
        plt.subplot(3, 2, idx + 1)
        plt.plot(actual[:PLOT_POINTS], label="Actual")
        plt.plot(predicted[:PLOT_POINTS], label="Predicted")
        plt.title(feature)
        plt.legend()

    plt.tight_layout()
    plt.savefig("model_evaluation_plot.png")
    plt.close()

    print("\nEvaluation finished")
    print("Saved plot: model_evaluation_plot.png")


if __name__ == "__main__":
    evaluate()
