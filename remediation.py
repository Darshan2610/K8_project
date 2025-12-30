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


import time
import subprocess
from typing import Dict

# ================= CONFIG =================

LOG_FILE = "remediation.log"

DRY_RUN = False  # True → no kubectl execution
DEFAULT_NAMESPACE = "default"
DEFAULT_DEPLOYMENT = "nginx"

COOLDOWN_SEC = 60  # per feature cooldown

# ================= REMEDIATION PLAN =================

REMEDIATION_PLAN: Dict[str, Dict] = {
    "cpu_allocation_efficiency": {
        "action": "scale",
        "delta": 1,
        "max_replicas": 10,
    },
    "memory_allocation_efficiency": {
        "action": "scale",
        "delta": 1,
        "max_replicas": 10,
    },
    "node_cpu_usage": {
        "action": "scale",
        "delta": 1,
        "max_replicas": 10,
    },
    "node_memory_usage": {
        "action": "scale",
        "delta": 1,
        "max_replicas": 10,
    },
    "network_latency": {
        "action": "restart",
    },
    "node_temperature": {
        "action": "alert",
    },
}

# Track last remediation per feature
_last_action_ts: Dict[str, float] = {}

# ================= LOGGING =================


def log_remediation(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)

    # Force UTF-8 to avoid Windows encoding crashes
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ================= KUBECTL HELPERS =================


def _kubectl(cmd):
    if DRY_RUN:
        log_remediation("[DRY-RUN] " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _get_replicas(namespace: str, deployment: str) -> int:
    out = subprocess.check_output(
        [
            "kubectl",
            "get",
            "deployment",
            deployment,
            "-n",
            namespace,
            "-o",
            "jsonpath={.spec.replicas}",
        ]
    )
    return int(out.decode())


# ================= REMEDIATION CORE =================


def simple_remediation(feature: str, current_value):
    """
    Executes remediation ONLY if:
    - feature exists in REMEDIATION_PLAN
    - cooldown has passed
    """

    if feature not in REMEDIATION_PLAN:
        log_remediation(f"No remediation defined for {feature}")
        return

    now = time.time()
    last = _last_action_ts.get(feature, 0)

    if now - last < COOLDOWN_SEC:
        log_remediation(f"{feature} in cooldown, skipping remediation")
        return

    _last_action_ts[feature] = now

    plan = REMEDIATION_PLAN[feature]
    action = plan["action"]

    # -------- SCALE --------
    if action == "scale":
        namespace = DEFAULT_NAMESPACE
        deployment = DEFAULT_DEPLOYMENT

        current = _get_replicas(namespace, deployment)
        delta = plan.get("delta", 1)
        max_rep = plan.get("max_replicas", current + delta)

        new = min(current + delta, max_rep)

        log_remediation(
            f"Scaling {namespace}/{deployment}: {current} -> {new} "
            f"(triggered by {feature})"
        )

        _kubectl(
            [
                "kubectl",
                "scale",
                "deployment",
                deployment,
                "-n",
                namespace,
                f"--replicas={new}",
            ]
        )

    # -------- RESTART --------
    elif action == "restart":
        namespace = DEFAULT_NAMESPACE

        pod = subprocess.check_output(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                namespace,
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
        ).decode()

        log_remediation(f"Restarting pod {pod} due to {feature}")

        _kubectl(["kubectl", "delete", "pod", pod, "-n", namespace])

    # -------- ALERT ONLY --------
    elif action == "alert":
        log_remediation(f"ALERT: {feature} exceeded safe threshold")

    else:
        log_remediation(f"Unknown action '{action}' for {feature}")
