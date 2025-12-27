import time
from collections import deque, defaultdict

import numpy as np
import joblib
import tensorflow as tf

try:
    from kubernetes import client as k8s_client, config as k8s_config

    K8S_AVAILABLE = True
except Exception:
    K8S_AVAILABLE = False

try:
    # reuse remediation logging and actions if available
    from remediation import simple_remediation, log_remediation
except Exception:
    simple_remediation = None


"""
k8s_monitor.py
--------------

Runtime watcher that:
- loads the CPU+memory Transformer model (pods_k8s_model.keras)
- loads the scaler (pods_scaler.pkl)
- periodically fetches CPU / memory usage for pods from the metrics API
  (requires metrics-server in your kind cluster)
- maintains a rolling window of recent metrics per pod
- performs multi-step (e.g. 5-step) prediction for each pod
- prints warnings if the predicted CPU / memory usage looks risky

This file is intentionally conservative: it ONLY monitors and prints
risk information. Actual remediation should be delegated to the
functions already defined in remediation.py (or extended later).
"""


MODEL_PATH = "pods_k8s_model.keras"
SCALER_PATH = "pods_scaler.pkl"

SEQ_LEN = 20  # must match pods_model.py
HORIZON_STEPS = 5  # must match pods_model.py

POLL_INTERVAL_SEC = 30
TARGET_NAMESPACE = "default"


def load_model_and_scaler():
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, scaler


def ensure_k8s_config():
    if not K8S_AVAILABLE:
        raise RuntimeError(
            "kubernetes Python client not installed. "
            "Install with: pip install kubernetes"
        )
    try:
        k8s_config.load_kube_config()
    except Exception:
        # useful when running inside the cluster
        k8s_config.load_incluster_config()


def fetch_pod_metrics(namespace="default"):
    """
    Fetch CPU and memory metrics for pods via metrics.k8s.io API.
    Requires metrics-server to be installed and working in the cluster.

    Returns dict:
        {
          "pod-name": {
             "cpu_mcores": <float>,    # millicores
             "mem_bytes": <float>,     # bytes
          },
          ...
        }
    """
    api = k8s_client.CustomObjectsApi()
    group = "metrics.k8s.io"
    version = "v1beta1"
    plural = "pods"

    try:
        resp = api.list_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to query metrics.k8s.io (is metrics-server installed and working?): "
            f"{e}"
        )

    result = {}
    items = resp.get("items", [])
    for item in items:
        pod_name = item["metadata"]["name"]
        containers = item.get("containers", [])
        cpu_mcores = 0.0
        mem_bytes = 0.0
        for c in containers:
            usage = c.get("usage", {})
            cpu_str = usage.get("cpu", "0")
            mem_str = usage.get("memory", "0")

            # cpu like "10m" or "0.1"
            if cpu_str.endswith("m"):
                cpu_mcores += float(cpu_str[:-1])
            else:
                # assume cores; convert to millicores
                try:
                    cpu_mcores += float(cpu_str) * 1000.0
                except ValueError:
                    pass

            # memory like "128Mi", "256Ki", etc.
            factor = 1.0
            if mem_str.endswith("Ki"):
                factor = 1024.0
                mem_val = mem_str[:-2]
            elif mem_str.endswith("Mi"):
                factor = 1024.0**2
                mem_val = mem_str[:-2]
            elif mem_str.endswith("Gi"):
                factor = 1024.0**3
                mem_val = mem_str[:-2]
            else:
                mem_val = mem_str

            try:
                mem_bytes += float(mem_val) * factor
            except ValueError:
                pass

        result[pod_name] = {
            "cpu_mcores": cpu_mcores,
            "mem_bytes": mem_bytes,
        }
    return result


def metric_vector(cpu_mcores, mem_bytes):
    """
    Map raw CPU/memory values into a 2-D feature vector for the model.
    The scaler will put them into the [0,1] range according to the
    training distribution.
    """
    return np.array([cpu_mcores, mem_bytes], dtype=np.float32)


def run_monitor_loop():
    model, scaler = load_model_and_scaler()
    ensure_k8s_config()

    # per-pod rolling windows of raw metrics
    history = defaultdict(lambda: deque(maxlen=SEQ_LEN))

    print(
        f"Starting Kubernetes pod monitor in namespace '{TARGET_NAMESPACE}' "
        f"(poll interval = {POLL_INTERVAL_SEC}s, horizon = {HORIZON_STEPS} steps)"
    )

    while True:
        try:
            pods_metrics = fetch_pod_metrics(namespace=TARGET_NAMESPACE)
        except Exception as e:
            print(str(e))
            print("Sleeping before retry...")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for pod_name, metrics in pods_metrics.items():
            vec = metric_vector(metrics["cpu_mcores"], metrics["mem_bytes"])
            history[pod_name].append(vec)

            if len(history[pod_name]) < SEQ_LEN:
                print(
                    f"Collecting history for {pod_name}: {len(history[pod_name])}/{SEQ_LEN}"
                )
                continue

            # build input for model: shape (1, SEQ_LEN, 2)
            hist_arr = np.array(history[pod_name], dtype=np.float32)
            # scale with same scaler used in training
            scaled_hist = scaler.transform(hist_arr)
            X = np.expand_dims(scaled_hist, axis=0)

            y_pred_flat = model.predict(X, verbose=0)[0]  # shape (HORIZON_STEPS * 2,)
            y_pred = y_pred_flat.reshape(HORIZON_STEPS, 2)
            # inverse transform each predicted step back to original units
            y_pred_inv = scaler.inverse_transform(y_pred)

            # Very simple risk heuristic:
            # - if any future CPU usage > cpu_mcores * 1.5 or an absolute threshold
            # - if any future memory > mem_bytes * 1.5 or an absolute threshold
            future_cpu = y_pred_inv[:, 0]
            future_mem = y_pred_inv[:, 1]

            cpu_now = metrics["cpu_mcores"]
            mem_now = metrics["mem_bytes"]

            cpu_abs_threshold = 800.0  # 800m = 0.8 core
            mem_abs_threshold = 512 * 1024 * 1024  # 512Mi

            cpu_risk = np.any(
                (future_cpu > cpu_now * 1.5) | (future_cpu > cpu_abs_threshold)
            )
            mem_risk = np.any(
                (future_mem > mem_now * 1.5) | (future_mem > mem_abs_threshold)
            )

            if cpu_risk or mem_risk:
                # Build a clean, timestamp-free message; log_remediation will add timestamp.
                risk_msg = (
                    f"Pod {pod_name} → RISK "
                    f"(CPU={cpu_now:.1f}m, MEM={int(mem_now/1024/1024)}Mi). "
                    f"Suggested action: increase replicas (see REMEDIATION_PLAN)."
                )
                print("-" * 60)

                if simple_remediation is not None:
                    try:
                        log_remediation(risk_msg)
                    except Exception:
                        print(risk_msg)
                else:
                    print(risk_msg)

                # Invoke remediation logic using the same plan as remediation.py.
                if simple_remediation is not None:
                    dummy_indices = np.array([0])
                    if cpu_risk:
                        simple_remediation("node_cpu_usage", dummy_indices)
                    if mem_risk:
                        simple_remediation("node_memory_usage", dummy_indices)
            else:
                ok_msg = (
                    f"Pod {pod_name} → OK "
                    f"(CPU={cpu_now:.1f}m, MEM={int(mem_now/1024/1024)}Mi)"
                )
                print(ok_msg)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    run_monitor_loop()
