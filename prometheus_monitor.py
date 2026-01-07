import os
import time
from collections import defaultdict, deque
from typing import Dict
import requests
import numpy as np
import joblib
import tensorflow as tf
from remediation import simple_remediation, log_remediation, REMEDIATION_PLAN
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Config ---
HISTORY_CSV = "prom_history.csv"
SEQ_LEN = 20
HORIZON_STEPS = 5
MODEL_PATH = "models\multi_k8s_model.keras"
SCALER_PATH = "models\multi_scaler.pkl"
SCALE_COOLDOWN_SEC = 60*2
RETRAIN_INTERVAL_SEC = 60
SAMPLE_INTERVAL_SEC = 10

# Seven pod-scoped features (names kept to match your CSV header)
FEATURES = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "network_latency",
    "node_cpu_usage",
    "node_memory_usage",
    "node_temperature",
]


def append_history(timestamp, metrics):
    file_exists = os.path.isfile(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp"] + list(metrics.keys()))
        writer.writerow([timestamp] + list(metrics.values()))


def prom_query_instant(expr: str) -> float:
    try:
        resp = requests.get(
            "http://localhost:9090/api/v1/query", params={"query": expr}, timeout=5
        )
        resp.raise_for_status()
        result = resp.json().get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except Exception:
        return 0.0


def fetch_features_for_workload(namespace: str, deployment: str) -> np.ndarray:
    # Pod-level CPU usage (cores)
    cpu_usage = prom_query_instant(
        f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{deployment}-.*", container!="POD", container!=""}}[1m]))'
    )

    # Pod-level memory usage (bytes)
    mem_usage = prom_query_instant(
        f'sum(container_memory_working_set_bytes{{namespace="{namespace}", pod=~"{deployment}-.*"}})'
    )

    # Pod-level resource requests
    cpu_req = (
        prom_query_instant(
            f'sum(kube_pod_container_resource_requests_cpu_cores{{namespace="{namespace}", pod=~"{deployment}-.*"}})'
        )
        or 1e-6
    )
    mem_req = (
        prom_query_instant(
            f'sum(kube_pod_container_resource_requests_memory_bytes{{namespace="{namespace}", pod=~"{deployment}-.*"}})'
        )
        or 1e-6
    )

    # cpu_eff = 0.0 if cpu_req < 1e-5 else cpu_usage / cpu_req
    # mem_eff = 0.0 if mem_req < 1e-5 else mem_usage / mem_req
    # Ensure CPU request is at least 0.01 cores
    cpu_eff = cpu_usage / max(cpu_req, 0.01)

    # Ensure memory request is at least 1 MB
    mem_eff = mem_usage / max(mem_req, 1024 * 1024)

    # Pod-level latency (p95)
    latency_p95 = prom_query_instant(
        f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{namespace="{namespace}", pod=~"{deployment}-.*"}}[1m])) by (le))'
    )

    # Pod CPU and memory as percentage of node capacity
    total_cpu = prom_query_instant("sum(machine_cpu_cores)") or 1.0
    total_mem = prom_query_instant("sum(machine_memory_bytes)") or 1.0
    pod_cpu_pct = (cpu_usage / total_cpu) * 100.0
    pod_mem_pct = (mem_usage / total_mem) * 100.0

    pod_temp = 0.0  # placeholder

    # Clip values
    cpu_eff = float(np.clip(cpu_eff, 0.0, 10.0))
    mem_eff = float(np.clip(mem_eff, 0.0, 10.0))
    latency_p95 = float(np.clip(latency_p95, 0.0, 1e6))
    pod_cpu_pct = float(np.clip(pod_cpu_pct, 0.0, 100.0))
    pod_mem_pct = float(np.clip(pod_mem_pct, 0.0, 100.0))

    return np.array(
        [cpu_eff, mem_eff, latency_p95, pod_cpu_pct, pod_mem_pct, pod_temp],
        dtype=np.float32,
    )


def safe_load_scaler():
    """Load scaler if compatible; otherwise try to re-fit from HISTORY_CSV or create a fresh scaler."""
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(
                FEATURES
            ):
                return scaler
            else:
                log_remediation(
                    "Saved scaler feature count mismatch. Attempting to re-fit from history."
                )
        except Exception:
            log_remediation(
                "Failed to load saved scaler. Will attempt to fit a new one."
            )
    # Try to fit from HISTORY_CSV
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            # Ensure required columns exist
            missing = [c for c in FEATURES if c not in df.columns]
            if not missing:
                X = df[FEATURES].values.astype(np.float32)
                scaler = MinMaxScaler()
                scaler.fit(X)
                joblib.dump(scaler, SCALER_PATH)
                log_remediation("Fitted new scaler from HISTORY_CSV.")
                return scaler
            else:
                log_remediation(
                    f"History CSV missing columns: {missing}. Creating fresh scaler."
                )
        except Exception as e:
            log_remediation(f"Error reading HISTORY_CSV for scaler fit: {e}")
    # Fallback: create and save an unfitted scaler (will be fitted during retrain)
    scaler = MinMaxScaler()
    joblib.dump(scaler, SCALER_PATH)
    log_remediation("Initialized fresh scaler (will be fitted during retrain).")
    return scaler


def safe_load_model():
    """Load model if compatible with current FEATURES and SEQ_LEN. If incompatible, return None."""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # Check input shape: (batch, timesteps, features)
            input_shape = model.input_shape  # e.g., (None, SEQ_LEN, n_features)
            if input_shape is None:
                return None
            # Some Keras models may have nested shapes; check last two dims
            if (
                len(input_shape) >= 3
                and input_shape[1] == SEQ_LEN
                and input_shape[2] == len(FEATURES)
            ):
                return model
            else:
                log_remediation(
                    "Saved model input shape mismatch. Will retrain model when enough history is available."
                )
                return None
        except Exception:
            log_remediation("Failed to load saved model. Will retrain when possible.")
            return None
    return None


def load_model_and_scaler():
    scaler = safe_load_scaler()
    model = safe_load_model()
    return model, scaler


def retrain_model_from_history(history: Dict[str, deque]):
    X_list, Y_list = [], []
    for h in history.values():
        arr = np.array(h, dtype=np.float32)
        if len(arr) >= SEQ_LEN + HORIZON_STEPS:
            for i in range(len(arr) - SEQ_LEN - HORIZON_STEPS + 1):
                X_list.append(arr[i : i + SEQ_LEN])
                Y_list.append(arr[i + SEQ_LEN : i + SEQ_LEN + HORIZON_STEPS].flatten())
    if not X_list:
        log_remediation("Not enough data to retrain model.")
        return

    X_train = np.array(X_list)  # shape (samples, SEQ_LEN, n_features)
    Y_train = np.array(Y_list)  # shape (samples, HORIZON_STEPS * n_features)

    # Fit scaler on training data (reshape to 2D)
    X_2d = X_train.reshape(-1, X_train.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(X_2d)
    joblib.dump(scaler, SCALER_PATH)

    # Transform X_train
    X_train_scaled = scaler.transform(X_2d).reshape(X_train.shape)

    # Build a simple LSTM model matching the new feature count
    n_features = X_train.shape[2]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQ_LEN, n_features)),
            tf.keras.layers.LSTM(64, activation="tanh"),
            tf.keras.layers.Dense(HORIZON_STEPS * n_features),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, Y_train, epochs=5, batch_size=16, verbose=0)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log_remediation("Retraining complete: new model and scaler saved.")


def pretty_table_output(key, feats_now, y_pred_inv):
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | {key} ===")
    print("+---------------------------------------+------------+--------------+")
    print("| {:<37} | {:>10} | {:>12} |".format("Feature", "Current", "Pred (5min)"))
    print("+---------------------------------------+------------+--------------+")
    for i, feat in enumerate(FEATURES):
        cur = float(feats_now[i])
        pred = float(y_pred_inv[-1, i]) if y_pred_inv is not None else 0.0
        print("| {:<37} | {:>10.2f} | {:>12.2f} |".format(feat, cur, pred))
    print("+---------------------------------------+------------+--------------+")


def run_monitor_loop(target_namespace="default", target_deployment="nginx"):
    model, scaler = load_model_and_scaler()
    history = defaultdict(lambda: deque(maxlen=SEQ_LEN + HORIZON_STEPS + 200))
    last_scale_ts, last_retrain_ts = {}, 0
    key = f"{target_namespace}/{target_deployment}"

    log_remediation(f"Starting Prometheus monitor for {key}")

    try:
        while True:
            feats = fetch_features_for_workload(target_namespace, target_deployment)
            metrics_dict = dict(zip(FEATURES, feats))
            append_history(time.strftime("%Y-%m-%d %H:%M:%S"), metrics_dict)
            history[key].append(feats)

            # Warm-up until we have SEQ_LEN samples
            if len(history[key]) < SEQ_LEN:
                print(f"Collecting history: {len(history[key])}/{SEQ_LEN}")
                time.sleep(SAMPLE_INTERVAL_SEC)
                continue

            # If scaler is not fitted yet, try to fit from history now
            if not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ != len(
                FEATURES
            ):
                try:
                    df = pd.read_csv(HISTORY_CSV)
                    if all(c in df.columns for c in FEATURES):
                        X = df[FEATURES].values.astype(np.float32)
                        scaler = MinMaxScaler()
                        scaler.fit(X)
                        joblib.dump(scaler, SCALER_PATH)
                        log_remediation(
                            "Fitted scaler from HISTORY_CSV during runtime."
                        )
                    else:
                        log_remediation(
                            "Cannot fit scaler from HISTORY_CSV: missing columns."
                        )
                except Exception as e:
                    log_remediation(f"Error fitting scaler from HISTORY_CSV: {e}")

            # If model is not available (shape mismatch or missing), attempt to retrain if enough history
            if model is None:
                # If we have enough history to retrain, call retrain
                if len(history[key]) >= SEQ_LEN + HORIZON_STEPS:
                    retrain_model_from_history(history)
                    model = safe_load_model()
                    scaler = safe_load_scaler()
                    if model is None:
                        log_remediation(
                            "Model still not available after retrain attempt."
                        )
                else:
                    log_remediation(
                        "Model not available; collecting more history for retraining."
                    )
                    time.sleep(SAMPLE_INTERVAL_SEC)
                    continue

            # Prepare model input
            hist_arr = np.array(list(history[key])[-SEQ_LEN:])
            # Ensure scaler is fitted before transform
            if not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ != len(
                FEATURES
            ):
                log_remediation(
                    "Scaler not fitted for current feature set; skipping prediction this cycle."
                )
                time.sleep(SAMPLE_INTERVAL_SEC)
                continue

            X = np.expand_dims(scaler.transform(hist_arr), axis=0)

            # Predict horizon and invert scaling
            try:
                y_pred = model.predict(X, verbose=0).reshape(
                    HORIZON_STEPS, len(FEATURES)
                )
                y_pred_inv = scaler.inverse_transform(y_pred)
                y_pred = np.clip(y_pred, 0, None)
            except Exception as e:
                log_remediation(
                    f"Prediction failed: {e}. Will attempt retrain next cycle."
                )
                y_pred_inv = None

            # Display table
            pretty_table_output(
                key,
                feats,
                (
                    y_pred_inv
                    if y_pred_inv is not None
                    else np.zeros((HORIZON_STEPS, len(FEATURES)))
                ),
            )

            # --- Auto-remediation (use cpu_allocation_efficiency) ---
            # --- Auto-remediation using REMEDIATION_PLAN ---
            for feat in FEATURES:
                value_now = feats[FEATURES.index(feat)]
                future_max = 0.0
                if y_pred_inv is not None:
                    future_vals = np.clip(y_pred_inv[:, FEATURES.index(feat)], 0, None)
                    future_max = np.max(future_vals)

                THRESHOLDS = {
                    "cpu_allocation_efficiency": 11,
                    "memory_allocation_efficiency": 12,
                    "disk_io": 1e9,  # example: 1 GB/s
                    "network_latency": 500,  # example: 500 ms
                    "node_cpu_usage": 4,  # percent
                    "node_memory_usage": 2,  # percent
                    "node_temperature": 85,  # degrees C
                }

                threshold = THRESHOLDS.get(feat, None)
                if threshold is None:
                    continue

                current_stress = value_now > threshold
                predicted_stress = future_max > threshold
                stress_detected = current_stress or predicted_stress

                if stress_detected:
                    last_scale = last_scale_ts.get(feat, 0)

                    if time.time() - last_scale > SCALE_COOLDOWN_SEC:
                        last_scale_ts[feat] = time.time()

                        log_remediation(
                            f"{key}: Remediation triggered | feature={feat} | "
                            f"now={value_now:.2f} | pred_max={future_max:.2f} | "
                            f"threshold={threshold} | "
                            f"current_stress={current_stress} | predicted_stress={predicted_stress}"
                        )

                        simple_remediation(feat, np.array([value_now]))
                    else:
                        log_remediation(
                            f"{key} | {feat} in cooldown, remediation skipped"
                        )

            # Periodic retrain
            if time.time() - last_retrain_ts > RETRAIN_INTERVAL_SEC:
                retrain_model_from_history(history)
                last_retrain_ts = time.time()
                # reload model and scaler after retrain
                model = safe_load_model()
                scaler = safe_load_scaler()

            time.sleep(SAMPLE_INTERVAL_SEC)

    except KeyboardInterrupt:
        log_remediation("Monitor interrupted by user. Exiting.")


if __name__ == "__main__":
    run_monitor_loop()
