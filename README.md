## Kubernetes Pod Behavior Prediction & Auto-Remediation

This project builds an AIOps-style system that:

- **Monitors Kubernetes workloads** (pods/deployments) in a kind cluster.
- **Predicts the next few minutes of resource usage** using deep learning (LSTM/Transformer).
- **Detects risk conditions** (e.g., future CPU overload).
- **Automatically remediates** issues (scales deployments, restarts pods, etc.).
- **Visualizes metrics and behavior** with Prometheus + Grafana.

The core use case here is a `nginx` deployment running in the `default` namespace on a local **kind** cluster.

---

kubectl scale deployment nginx --replicas=1

## 1. Quickstart – Commands Only


1. **(Once) Train the 7‑metric model**

   ```powershell
   cd "C:\Users\Darshan\Downloads\major projectt (2)\major projectt\K8s project"
   python multi_metric_model.py
   ```

2. **Ensure cluster, nginx deployment, and monitoring stack are running**

   ```powershell
   kubectl get deploy -A
   kubectl get pods -n monitoring
   ```

3. **(Optional) Open Prometheus & Grafana UIs**

   ```powershell
   kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090
   kubectl port-forward -n monitoring svc/kps-grafana 3000:80
   ```

4. **Start the live monitor (prediction + remediation)**

   ```powershell
   cd "C:\Users\Darshan\Downloads\major projectt (2)\major projectt\K8s project"
   python prometheus_monitor.py
   ```

5. **In another terminal, start stress on nginx to trigger scaling**

   ```powershell
   cd "C:\Users\Darshan\Downloads\major projectt (2)\major projectt\K8s project"
   python stress_pod.py
   ```

6. **Watch behavior**

   - Terminal with `prometheus_monitor.py` → predictions + RISK logs.
   - `remediation_log.txt` → remediation actions taken.
   - Grafana (optional) → CPU/memory graphs and replica count.

If you want to understand _how_ it works, read the sections below.

---

## 2. Project Structure (key files)

- `kubernetes_performance_metrics_dataset.csv`  
  Historical dataset of 7 Kubernetes/node metrics used for offline training.

- `multi_metric_model.py`  
  Trains a **7-metric Transformer model** for time-series forecasting:

  - Uses 7 features:
    - `cpu_allocation_efficiency`
    - `memory_allocation_efficiency`
    - `disk_io`
    - `network_latency`
    - `node_cpu_usage`
    - `node_memory_usage`
    - `node_temperature`
  - History window: 20 time steps.
  - Horizon: next 5 steps (assume 1 step ≈ 1 minute).
  - Outputs:
    - `multi_scaler.pkl`
    - `multi_k8s_model.keras`

- `prometheus_monitor.py`  
  **Live monitor** that:

  - Every 60 seconds:
    - Queries Prometheus for the 7 metrics for a given workload (e.g. `default/nginx`).
    - Maintains a 20-point history (20 minutes).
    - Uses `multi_k8s_model.keras` + `multi_scaler.pkl` to predict the next 5 minutes.
    - Detects CPU risk:
      - Current CPU non-trivial.
      - Predicted CPU grows > 1.5× current peak.
    - Estimates required replicas based on predicted CPU and a per-pod capacity.
    - Calls `simple_remediation("node_cpu_usage", ...)` from `remediation.py` to scale.
  - Uses PromQL expressions that approximate the 7 CSV metrics.

- `remediation.py`  
  Contains the **remediation engine** and an offline anomaly pipeline:

  - `REMEDIATION_PLAN` defines how to respond to anomalies per feature:
    - `node_cpu_usage` → `action: "scale"` with namespace + deployment + scale deltas.
    - `node_memory_usage` → `action: "restart_pod"`.
    - `node_temperature` → `action: "cordon_node"`.
  - `simple_remediation(feature, indices)`:
    - For `node_cpu_usage`, reads current replicas via Kubernetes API or `kubectl`.
    - Scales the deployment with `kubectl scale` honoring:
      - `DRY_RUN`
      - `REMEDIATE`
      - `scale_delta`, `max_delta`.
  - `log_remediation(msg)` logs to stdout and `remediation_log.txt`.
  - `run_offline_training_and_remediation()` (when run as `python remediation.py`):
    - Trains a Transformer on CSV.
    - Detects anomalies via MAD.
    - Calls `simple_remediation` on historical anomalies.
    - Produces `prediction_plot_with_anomalies.png`.

- `k8s_monitor.py`  
  Earlier version of a live monitor that uses the metrics API / metrics-server (not Prometheus). Kept for reference; the **Prometheus-based** path is preferred.

- `pods_model.py`, `pods_k8s_model.keras`, `pods_scaler.pkl`  
  Prototype 2-metric (CPU + memory) model; now superseded by the 7-metric `multi_metric_model.py`. Still usable if you want a simpler live monitor.

- `stress_pod.py`  
  Utility to **generate CPU and memory load** on an existing pod:
  - Finds a pod by label (default `app=nginx` in `default` namespace).
  - Runs a shell script inside the pod that:
    - Allocates ~256Mi of memory.
    - Busy-loops for `STRESS_SECONDS` (default 300 seconds).
  - Used to **simulate overload** and trigger the monitoring + remediation pipeline.

---

## 3. Prerequisites

### 3.1 Local environment

- OS: Windows (PowerShell), but commands are standard kubectl/Helm/Python.
- Python 3.10+ (venv recommended).
- `pip install -r requirements` equivalent (at minimum):
  - `tensorflow` / `tensorflow-cpu`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `kubernetes`
  - `requests`

> You already have a `venv` in the repo; activate it before running Python scripts.

### 3.2 Kubernetes (kind) cluster

- A local **kind** cluster running:
  - The `nginx` deployment in `default` namespace.

Check:

```powershell
kubectl get deploy -A
```

You should see:

- `NAMESPACE: default`
- `NAME: nginx`

### 3.3 Monitoring stack (Prometheus + Grafana)

Install kube-prometheus-stack via Helm (once):

```powershell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kps prometheus-community/kube-prometheus-stack `
  --namespace monitoring --create-namespace
```

Check pods:

```powershell
kubectl get pods -n monitoring
```

You should see `kps-kube-prometheus-stack-prometheus-*`, `kps-grafana-*`, etc.

### 3.4 Port-forward Prometheus & Grafana (optional, for UI)

Prometheus:

```powershell
kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090
```

Grafana:

```powershell
kubectl port-forward -n monitoring svc/kps-grafana 3000:80
```

- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`

Add Prometheus as a data source in Grafana (Configuration → Data sources → Add data source → Prometheus).

---

## 4. Offline model training (explanation)

- `multi_metric_model.py`:
  - Reads the CSV dataset.
  - Scales the 7 features with `MinMaxScaler`.
  - Builds (20‑step history → 5‑step future) sequences.
  - Trains a Transformer and saves `multi_scaler.pkl` + `multi_k8s_model.keras`.
- `remediation.py` (when run directly):
  - Trains a simpler Transformer on the CSV.
  - Uses MAD to detect anomalies on historical data.
  - Calls `simple_remediation` for each anomalous feature.
  - Writes `prediction_plot_with_anomalies.png` and logs actions.

(Run commands from the Quickstart section when you actually want to execute these.)

---

## 5. Live Prometheus-backed monitoring & remediation (explanation)

This is the **main demo path**.

### 4.1 Configure remediation plan

In `remediation.py`, ensure `REMEDIATION_PLAN` and flags are set appropriately:

- Example:

```python
REMEDIATION_PLAN = {
    "node_cpu_usage": {
        "action": "scale",
        "namespace": "default",
        "deployment": "nginx",
        "scale_delta": 1,
        "max_delta": 3,
    },
    "node_memory_usage": {"action": "restart_pod", "namespace": "default"},
    "node_temperature": {"action": "cordon_node"},
}
```

- Control flags:

```python
DRY_RUN = False   # set True to log only, False to actually apply changes
REMEDIATE = True  # must be True to allow real remediation
```

> For safety, you can demo first with `DRY_RUN = True` and then switch to `False` once you trust the behavior.

What `prometheus_monitor.py` does every 60 seconds:

1. Uses PromQL to fetch a 7-dimensional feature vector for the workload (namespace/deployment taken from `REMEDIATION_PLAN["node_cpu_usage"]`).
2. Appends it to a per-workload history deque of length 20 (20 minutes).
3. Once 20 points exist:
   - Scales them via `multi_scaler.pkl`.
   - Calls `multi_k8s_model.keras` to predict the next 5 minutes for all 7 metrics.
   - Extracts predicted CPU usage (`node_cpu_usage`) over the next 5 steps.
   - Treats it as **CPU risk** if:
     - Current CPU > small threshold, and
     - Peak predicted CPU > 1.5 × current CPU.
4. Logs:
   - Current CPU and 5-step future CPU trace.
   - If risk:
     - Estimates required replicas based on a per-pod capacity.
     - Honors a scaling cooldown per workload (5 minutes).
     - Calls `simple_remediation("node_cpu_usage", ...)` to scale the deployment.
     - All actions and results recorded in `remediation_log.txt`.

Leave this running during your demo (see Quickstart for exact commands).

---

## 6. Stressing the workload to simulate risk (explanation)

`stress_pod.py` artificially generates load on the nginx pod:

- Finds a pod by label:
  - Namespace: `default`
  - Label selector: `app=nginx` (see `NAMESPACE` and `LABEL_SELECTOR` at the top of the file).
- Executes a shell script inside the pod that:
  - Writes ~256Mi to `/dev/shm/stress_mem` to consume memory.
  - Busy-loops until `now + STRESS_SECONDS` using `date +%s` in a `while` loop.
- Default duration:

```python
STRESS_SECONDS = 300  # 5 minutes
```

You can adjust `STRESS_SECONDS` to change stress duration.

During stress:

- Prometheus metrics (CPU/memory) for the nginx pod should rise.
- `prometheus_monitor.py` should log increased `now CPU` and `future CPU`.
- When risk is detected and cooldown permits:
  - `simple_remediation` scales the nginx deployment up.
  - `remediation_log.txt` records:
    - Risk detections.
    - Scaling decisions.
    - Success/failure of `kubectl` or API calls.

---

## 7. Grafana dashboard guidelines (queries only)

With Grafana connected to Prometheus as a data source, you can build panels showing:

### 6.1 CPU usage for nginx (whole deployment)

In a Time series panel, Prometheus query (adjust metric name if needed based on Prometheus UI):

Preferred (if `container_cpu_usage_seconds_total` exists):

```promql
sum(
  rate(
    container_cpu_usage_seconds_total{
      namespace="default",
      pod=~"nginx-.*",
      image!=""
    }[5m]
  )
)
```

Fallback using `container_cpu_cfs_periods_total`:

```promql
sum(
  rate(
    container_cpu_cfs_periods_total{
      namespace="default",
      pod=~"nginx-.*",
      image!=""
    }[5m]
  )
)
```

### 6.2 Memory usage for nginx

```promql
sum(
  container_memory_working_set_bytes{
    namespace="default",
    pod=~"nginx-.*",
    image!=""
  }
)
```

Set panel unit to `bytes`.

### 6.3 Replicas for nginx (shows remediation effect)

```promql
kube_deployment_status_replicas{namespace="default", deployment="nginx"}
```

You should see step changes when remediation scales the deployment.

### 6.4 Other metrics (disk I/O, latency, temperature)

Examples (may need tuning based on your Prometheus metric set):

- Disk I/O:

```promql
sum(
  rate(node_disk_read_bytes_total[5m]) +
  rate(node_disk_written_bytes_total[5m])
)
```

- Network latency (95th percentile):

```promql
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
```

- Node temperature:

```promql
avg(node_thermal_zone_temp) * 0.001
```

This provides an end-to-end demonstration of:

- Historical modeling and anomaly detection.
- Live predictive monitoring using Prometheus data.
- Automated remediation actions to keep pods/nodes healthy.
