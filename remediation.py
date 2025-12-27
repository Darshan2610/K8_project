# import os
# import time
# import subprocess
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# try:
#     from kubernetes import client as k8s_client, config as k8s_config

#     K8S_AVAILABLE = True
# except Exception:
#     K8S_AVAILABLE = False

# FILE_PATH = "prom_history.csv"
# SEQ_LEN = 20
# MAD_K = 3.0
# FALLBACK_Q = 0.98
# MIN_PTS = 5

# DRY_RUN = False
# REMEDIATE = True

# REM_LOG = "remediation_log.txt"

# REMEDIATION_PLAN = {
#     "node_cpu_usage": {
#         "action": "scale",
#         "namespace": "default",
#         "deployment": "nginx",
#         "scale_delta": 1,
#         "max_delta": 3,
#     },
#     "node_memory_usage": {"action": "restart_pod", "namespace": "default"},
#     "node_temperature": {"action": "cordon_node"},
# }


# def now_ts():
#     return int(time.time())


# def log_remediation(msg):
#     ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     line = f"[{ts}] {msg}\n"
#     print(line.strip())
#     try:
#         with open(REM_LOG, "a") as f:
#             f.write(line)
#     except Exception:
#         pass


# def mad_based_threshold(res):
#     med = np.median(res)
#     mad = np.median(np.abs(res - med))
#     robust_std = 1.4826 * mad if mad > 0 else np.std(res) + 1e-8
#     thr = med + MAD_K * robust_std
#     idx = np.where(res > thr)[0]

#     if idx.size == 0:
#         qthr = np.quantile(res, FALLBACK_Q)
#         idx = np.where(res >= qthr)[0]
#         if idx.size < MIN_PTS:
#             idx = np.argsort(res)[-MIN_PTS:]
#     return np.sort(idx)


# def kubectl_run(cmd_list, dry_run=True):
#     if dry_run or not REMEDIATE:
#         log_remediation(f"[DRY-RUN] Would run: {' '.join(cmd_list)}")
#         return True, "dry-run"
#     try:
#         out = subprocess.check_output(cmd_list, stderr=subprocess.STDOUT).decode()
#         return True, out
#     except Exception as e:
#         return False, str(e)


# def scale_deployment_via_kubectl(deployment, namespace, replicas):
#     cmd = [
#         "kubectl",
#         "scale",
#         "deployment",
#         deployment,
#         "-n",
#         namespace,
#         f"--replicas={replicas}",
#     ]
#     return kubectl_run(cmd, dry_run=DRY_RUN)


# def delete_pod_via_kubectl(pod_name, namespace="default"):
#     cmd = ["kubectl", "delete", "pod", pod_name, "-n", namespace]
#     return kubectl_run(cmd, dry_run=DRY_RUN)


# def cordon_node_via_kubectl(node_name):
#     cmd = ["kubectl", "cordon", node_name]
#     return kubectl_run(cmd, dry_run=DRY_RUN)


# def choose_pod_for_restart(namespace="default"):
#     if not K8S_AVAILABLE:
#         return None
#     try:
#         v1 = k8s_client.CoreV1Api()
#         pods = v1.list_namespaced_pod(namespace)
#         if pods.items:
#             return pods.items[0].metadata.name
#     except Exception:
#         return None
#     return None


# def simple_remediation(feature, indices):
#     plan = REMEDIATION_PLAN.get(feature, {"action": "alert_only"})
#     action = plan.get("action", "alert_only")

#     indices_short = list(indices[:10]) if len(indices) > 0 else []
#     log_remediation(
#         f"Detected anomalies for '{feature}' at indices {indices_short} (total={len(indices)})"
#     )

#     if action == "scale":
#         ns = plan.get("namespace", "default")
#         dep = plan.get("deployment", "my-deployment")
#         delta = int(plan.get("scale_delta", 1))
#         max_delta = int(plan.get("max_delta", 3))

#         cur_replicas = None
#         try:
#             if K8S_AVAILABLE:
#                 k8s_cfg = k8s_config
#                 try:
#                     k8s_cfg.load_kube_config()
#                 except Exception:
#                     try:
#                         k8s_cfg.load_incluster_config()
#                     except Exception:
#                         pass
#                 api = k8s_client.AppsV1Api()
#                 cur_replicas = api.read_namespaced_deployment_scale(
#                     dep, ns
#                 ).spec.replicas
#             else:
#                 out = subprocess.check_output(
#                     [
#                         "kubectl",
#                         "get",
#                         "deployment",
#                         dep,
#                         "-n",
#                         ns,
#                         "-o",
#                         "jsonpath={.spec.replicas}",
#                     ],
#                     stderr=subprocess.STDOUT,
#                 ).decode()
#                 cur_replicas = int(out.strip())
#         except Exception:
#             cur_replicas = 1

#         new_replicas = max(1, min(cur_replicas + delta, cur_replicas + max_delta))
#         log_remediation(
#             f"Action: scale {ns}/{dep} from {cur_replicas} -> {new_replicas}"
#         )
#         ok, msg = scale_deployment_via_kubectl(dep, ns, new_replicas)
#         if ok:
#             log_remediation(f"Scale command accepted: {msg}")
#         else:
#             log_remediation(f"Scale failed: {msg}")
#         return ok

#     elif action == "restart_pod":
#         ns = plan.get("namespace", "default")
#         pod = choose_pod_for_restart(ns)
#         if pod is None:
#             log_remediation("No pod found to restart; notify human.")
#             return False
#         log_remediation(f"Action: restart pod {ns}/{pod}")
#         ok, msg = delete_pod_via_kubectl(pod, namespace=ns)
#         if ok:
#             log_remediation(f"Restart command accepted: {msg}")
#         else:
#             log_remediation(f"Restart failed: {msg}")
#         return ok

#     elif action == "cordon_node":
#         node_name = plan.get("node_name")
#         if not node_name:
#             log_remediation(
#                 "No node specified to cordon; please configure node_name in REMEDIATION_PLAN."
#             )
#             return False
#         log_remediation(f"Action: cordon node {node_name}")
#         ok, msg = cordon_node_via_kubectl(node_name)
#         if ok:
#             log_remediation(f"Cordon command accepted: {msg}")
#         else:
#             log_remediation(f"Cordon failed: {msg}")
#         return ok

#     else:
#         log_remediation(
#             f"ALERT: {feature} anomalies detected. Manual inspection recommended."
#         )
#         return False


# def run_offline_training_and_remediation():
#     """
#     Offline mode:
#     - trains a Transformer on the historical CSV
#     - detects anomalies using MAD
#     - runs remediation actions accordingly
#     - produces prediction_plot_with_anomalies.png
#     """
#     df = pd.read_csv(FILE_PATH)

#     features = [
#         "cpu_allocation_efficiency",
#         "memory_allocation_efficiency",
#         "disk_io",
#         "network_latency",
#         "node_cpu_usage",
#         "node_memory_usage",
#         "node_temperature",
#     ]

#     df = df[features].dropna().head(500)

#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(df)
#     joblib.dump(scaler, "scaler_k8s.pkl")

#     X, y = [], []
#     for i in range(SEQ_LEN, len(scaled)):
#         X.append(scaled[i - SEQ_LEN : i])
#         y.append(scaled[i])
#     X = np.array(X)
#     y = np.array(y)

#     inputs = layers.Input(shape=(SEQ_LEN, X.shape[2]))
#     attn = layers.MultiHeadAttention(num_heads=2, key_dim=X.shape[2])(inputs, inputs)
#     x = layers.Add()([inputs, attn])
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     ffn = layers.Dense(64, activation="relu")(x)
#     ffn = layers.Dense(X.shape[2])(ffn)
#     x = layers.Add()([x, ffn])
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
#     pooled = layers.GlobalAveragePooling1D()(x)
#     outputs = layers.Dense(X.shape[2])(pooled)
#     model = models.Model(inputs=inputs, outputs=outputs)

#     model.compile(optimizer="adam", loss="mse")
#     model.fit(X, y, epochs=10, batch_size=32, verbose=1)
#     model.save("transformer_k8s_model.keras")

#     pred_scaled = model.predict(X, verbose=0)
#     actual = scaler.inverse_transform(y)
#     predicted = scaler.inverse_transform(pred_scaled)

#     residuals = np.abs(actual - predicted)
#     anom_idx_by_feat = [
#         mad_based_threshold(residuals[:, i]) for i in range(len(features))
#     ]

#     any_anomaly = False
#     for i, feat in enumerate(features):
#         indices = anom_idx_by_feat[i]
#         if len(indices) > 0:
#             any_anomaly = True
#             simple_remediation(feat, indices)
#         else:
#             print(f"[OK] No anomalies found for {feat}")

#     if not any_anomaly:
#         print("[OK] No anomalies detected across all monitored features.")

#     plt.figure(figsize=(10, 14))
#     for i, feat in enumerate(features):
#         plt.subplot(len(features), 1, i + 1)
#         plt.plot(actual[:, i], label="Actual")
#         plt.plot(predicted[:, i], label="Predicted")

#         ai = anom_idx_by_feat[i]
#         plt.scatter(
#             ai,
#             actual[ai, i],
#             s=28,
#             color="red",
#             marker="o",
#             edgecolors="black",
#             linewidths=0.4,
#             zorder=6,
#             label="Anomaly",
#         )

#         plt.title(feat)
#         plt.xlabel("Sample")
#         plt.ylabel("Value")

#         handles, labels = plt.gca().get_legend_handles_labels()
#         seen, H, L = set(), [], []
#         for h, l in zip(handles, labels):
#             if l not in seen:
#                 H.append(h)
#                 L.append(l)
#                 seen.add(l)
#         plt.legend(H, L, loc="best")

#     plt.tight_layout()
#     plt.savefig("prediction_plot_with_anomalies.png")
#     plt.close()

#     print("Saved: prediction_plot_with_anomalies.png")
#     log_remediation("Script run complete.")


# if __name__ == "__main__":
#     run_offline_training_and_remediation()


# remediation.py
import os
import time
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Optional Kubernetes client
try:
    from kubernetes import client as k8s_client, config as k8s_config

    K8S_AVAILABLE = True
except Exception:
    K8S_AVAILABLE = False

# --- Config ---
FILE_PATH = "prom_history.csv"
SEQ_LEN = 5
MAD_K = 3.0
FALLBACK_Q = 0.98
MIN_PTS = 5

# Toggle dry-run for safe testing
DRY_RUN = False
REMEDIATE = True

REM_LOG = "remediation_log.txt"

# Remediation mapped to the 7-feature names used by the monitor/trainer
REMEDIATION_PLAN = {
    "cpu_allocation_efficiency": {
        "action": "scale",
        "namespace": "default",
        "deployment": "nginx",
        "scale_delta": 1,
        "max_delta": 3,
    },
    "memory_allocation_efficiency": {"action": "restart_pod", "namespace": "default"},
    "disk_io": {"action": "alert_only"},
    "network_latency": {"action": "alert_only"},
    "node_cpu_usage": {
        "action": "scale",
        "namespace": "default",
        "deployment": "nginx",
        "scale_delta": 1,
        "max_delta": 3,
    },
    "node_memory_usage": {"action": "restart_pod", "namespace": "default"},
    "node_temperature": {
        "action": "cordon_node",
        "node_name": None,
    },  # set node_name if needed
}


def now_ts():
    return int(time.time())


def log_remediation(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\n"
    print(line.strip())
    try:
        with open(REM_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass


def mad_based_threshold(res):
    med = np.median(res)
    mad = np.median(np.abs(res - med))
    robust_std = 1.4826 * mad if mad > 0 else (np.std(res) + 1e-8)
    thr = med + MAD_K * robust_std
    idx = np.where(res > thr)[0]

    if idx.size == 0:
        qthr = np.quantile(res, FALLBACK_Q)
        idx = np.where(res >= qthr)[0]
        if idx.size < MIN_PTS:
            idx = np.argsort(res)[-MIN_PTS:]
    return np.sort(idx)


def ensure_kube_config_loaded():
    if not K8S_AVAILABLE:
        return False
    try:
        try:
            k8s_config.load_kube_config()
        except Exception:
            k8s_config.load_incluster_config()
        return True
    except Exception:
        return False


def kubectl_run(cmd_list, dry_run=True):
    if dry_run or not REMEDIATE:
        log_remediation(f"[DRY-RUN] Would run: {' '.join(cmd_list)}")
        return True, "dry-run"
    try:
        out = subprocess.check_output(cmd_list, stderr=subprocess.STDOUT).decode()
        return True, out
    except Exception as e:
        return False, str(e)


def scale_deployment_via_kubectl(deployment, namespace, replicas):
    cmd = [
        "kubectl",
        "scale",
        "deployment",
        deployment,
        "-n",
        namespace,
        f"--replicas={replicas}",
    ]
    return kubectl_run(cmd, dry_run=DRY_RUN)


def delete_pod_via_kubectl(pod_name, namespace="default"):
    cmd = ["kubectl", "delete", "pod", pod_name, "-n", namespace]
    return kubectl_run(cmd, dry_run=DRY_RUN)


def cordon_node_via_kubectl(node_name):
    cmd = ["kubectl", "cordon", node_name]
    return kubectl_run(cmd, dry_run=DRY_RUN)


def choose_pod_for_restart(namespace="default"):
    # Try Kubernetes client first
    if K8S_AVAILABLE and ensure_kube_config_loaded():
        try:
            v1 = k8s_client.CoreV1Api()
            pods = v1.list_namespaced_pod(namespace)
            if pods.items:
                return pods.items[0].metadata.name
        except Exception:
            pass
    # Fallback to kubectl
    try:
        out = (
            subprocess.check_output(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.items[0].metadata.name}",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
        return out if out else None
    except Exception:
        return None


def get_current_replicas(deployment, namespace):
    # Try Kubernetes client first
    if K8S_AVAILABLE and ensure_kube_config_loaded():
        try:
            api = k8s_client.AppsV1Api()
            return api.read_namespaced_deployment_scale(
                deployment, namespace
            ).spec.replicas
        except Exception:
            pass
    # Fallback to kubectl
    try:
        out = (
            subprocess.check_output(
                [
                    "kubectl",
                    "get",
                    "deployment",
                    deployment,
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.spec.replicas}",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
        return int(out) if out else 1
    except Exception:
        return 1


def simple_remediation(feature, indices):
    plan = REMEDIATION_PLAN.get(feature, {"action": "alert_only"})
    action = plan.get("action", "alert_only")

    indices_short = list(indices[:10]) if len(indices) > 0 else []
    log_remediation(
        f"Anomaly detected in '{feature}' at indices {indices_short} (total={len(indices)})"
    )

    if action == "scale":
        ns = plan.get("namespace", "default")
        dep = plan.get("deployment", "nginx")
        delta = int(plan.get("scale_delta", 1))
        max_delta = int(plan.get("max_delta", 3))

        cur_replicas = get_current_replicas(dep, ns)
        new_replicas = max(1, min(cur_replicas + delta, cur_replicas + max_delta))
        log_remediation(
            f"Action: SCALE deployment {ns}/{dep} from {cur_replicas} → {new_replicas}"
        )
        ok, msg = scale_deployment_via_kubectl(dep, ns, new_replicas)
        log_remediation(
            ("Result: success → " if ok else "Result: failed → ") + str(msg)
        )
        return ok

    elif action == "restart_pod":
        ns = plan.get("namespace", "default")
        pod = choose_pod_for_restart(ns)
        if not pod:
            log_remediation(
                "Action: RESTART pod → No pod found, manual intervention required."
            )
            return False
        log_remediation(f"Action: RESTART pod {ns}/{pod}")
        ok, msg = delete_pod_via_kubectl(pod, namespace=ns)
        log_remediation(
            ("Result: success → " if ok else "Result: failed → ") + str(msg)
        )
        return ok

    elif action == "cordon_node":
        node_name = plan.get("node_name")
        if not node_name:
            log_remediation("Action: CORDON node → No node_name set in plan.")
            return False
        log_remediation(f"Action: CORDON node {node_name}")
        ok, msg = cordon_node_via_kubectl(node_name)
        log_remediation(
            ("Result: success → " if ok else "Result: failed → ") + str(msg)
        )
        return ok

    else:
        log_remediation(
            f"Action: ALERT ONLY for {feature} anomalies. Manual inspection recommended."
        )
        return False


def run_offline_training_and_remediation():
    """
    Offline mode:
    - trains a small Transformer on the historical CSV (same 7 features)
    - detects anomalies using MAD
    - runs remediation actions accordingly
    - produces prediction_plot_with_anomalies.png
    """
    df = pd.read_csv(FILE_PATH)

    features = [
        "cpu_allocation_efficiency",
        "memory_allocation_efficiency",
        "disk_io",
        "network_latency",
        "node_cpu_usage",
        "node_memory_usage",
        "node_temperature",
    ]

    df = df[features].dropna()
    if df.shape[0] < SEQ_LEN + 1:
        raise RuntimeError(
            f"Not enough rows for offline training. Need >= {SEQ_LEN + 1}, got {df.shape[0]}"
        )

    df = df.tail(800)  # keep last N for faster runs

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    joblib.dump(scaler, "scaler_k8s.pkl")

    # Build next-step prediction dataset
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN : i])
        y.append(scaled[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    inputs = layers.Input(shape=(SEQ_LEN, X.shape[2]))
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=max(1, X.shape[2] // 2))(
        inputs, inputs
    )
    x = layers.Add()([inputs, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn = layers.Dense(64, activation="relu")(x)
    ffn = layers.Dense(X.shape[2])(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(X.shape[2])(pooled)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=8, batch_size=32, verbose=1)
    model.save("transformer_k8s_model.keras")

    pred_scaled = model.predict(X, verbose=0)
    actual = scaler.inverse_transform(y)
    predicted = scaler.inverse_transform(pred_scaled)

    residuals = np.abs(actual - predicted)
    anom_idx_by_feat = [
        mad_based_threshold(residuals[:, i]) for i in range(len(features))
    ]

    any_anomaly = False
    for i, feat in enumerate(features):
        indices = anom_idx_by_feat[i]
        if len(indices) > 0:
            any_anomaly = True
            simple_remediation(feat, indices)
        else:
            print(f"[OK] No anomalies found for {feat}")

    if not any_anomaly:
        print("[OK] No anomalies detected across all monitored features.")

    # Plot
    plt.figure(figsize=(10, 14))
    for i, feat in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        plt.plot(actual[:, i], label="Actual")
        plt.plot(predicted[:, i], label="Predicted")

        ai = anom_idx_by_feat[i]
        if len(ai) > 0:
            plt.scatter(
                ai,
                actual[ai, i],
                s=28,
                color="red",
                marker="o",
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
                label="Anomaly",
            )

        plt.title(feat)
        plt.xlabel("Sample")
        plt.ylabel("Value")

        handles, labels = plt.gca().get_legend_handles_labels()
        seen, H, L = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                H.append(h)
                L.append(l)
                seen.add(l)
        plt.legend(H, L, loc="best")

    plt.tight_layout()
    plt.savefig("prediction_plot_with_anomalies.png")
    plt.close()

    print("Saved: prediction_plot_with_anomalies.png")
    log_remediation("Offline remediation run complete.")


if __name__ == "__main__":
    run_offline_training_and_remediation()
