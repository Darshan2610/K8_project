# 6. Implementation Approaches

## 6.1 Dataset Used

The Kubernetes performance metrics dataset represents a comprehensive collection of time-series data captured from a local kind cluster, specifically targeting the monitoring and prediction of pod behavior in a production-like environment. This dataset was curated to facilitate the development of automated remediation systems for Kubernetes workloads, focusing on proactive anomaly detection and resource optimization.

The dataset comprises historical metrics collected over multiple sessions from an nginx deployment running in the default namespace of a kind cluster. It includes 7 distinct features that capture various aspects of system performance and resource utilization:

- `cpu_allocation_efficiency`: Ratio of actual CPU usage to requested CPU resources, normalized to prevent over-allocation.
- `memory_allocation_efficiency`: Ratio of actual memory usage to requested memory resources, similarly normalized.
- `disk_io`: Aggregated disk read and write operations, measured in bytes per second.
- `network_latency`: 95th percentile of HTTP request durations, indicating network performance bottlenecks.
- `node_cpu_usage`: Percentage of total node CPU cores utilized by the workload.
- `node_memory_usage`: Percentage of total node memory utilized by the workload.
- `node_temperature`: Average thermal zone temperature of the node, scaled appropriately.

Each data point represents a snapshot taken at 1-minute intervals, with the dataset spanning approximately 2000 recent observations to ensure relevance and recency. The data was collected using Prometheus queries that approximate real-world Kubernetes metrics, ensuring compatibility with standard monitoring stacks. Prior to inclusion in the dataset, all metrics underwent preprocessing to handle missing values through linear interpolation and normalization to standardize feature scales, making them suitable for deep learning models.

This multi-dimensional dataset enables the training of predictive models that can forecast future resource demands and detect potential overload conditions, forming the foundation for an AIOps-style remediation system.

## 6.1.2 Experimental Setup

All experimental procedures, model training, and live monitoring processes were executed on a local development environment running Windows 11, utilizing Python 3.10+ within a virtual environment. The computational setup included an Intel Core i5 processor with 8 GB RAM, providing sufficient resources for training deep learning models without requiring cloud infrastructure.

The Kubernetes cluster was established using kind (Kubernetes in Docker), creating a lightweight, local cluster that mimics production environments. The monitoring stack consisted of Prometheus and Grafana, deployed via the kube-prometheus-stack Helm chart in a dedicated monitoring namespace. This setup allowed for real-time metric collection and visualization without external dependencies.

The experimental workflow followed a structured approach:

1. **Offline Training Phase**: Historical data from `kubernetes_performance_metrics_dataset.csv` was used to train predictive models.
2. **Live Monitoring Phase**: Trained models were deployed in a continuous monitoring loop, querying Prometheus every 60 seconds.
3. **Remediation Phase**: Detected anomalies triggered automated actions on the Kubernetes cluster.

Data partitioning allocated the most recent 2000 observations for training, with live data streams providing ongoing validation. This setup ensured a balance between historical learning and real-time adaptability.

## 6.1.3 Training Strategy

To achieve optimal predictive performance for multi-metric time-series forecasting, the framework employed a Transformer-based architecture designed for sequence-to-sequence prediction. The training strategy focused on capturing temporal dependencies across the 7-dimensional feature space while maintaining computational efficiency.

The model architecture consisted of:

- **Input Processing**: 20-time-step historical windows (approximately 20 minutes) of the 7 metrics.
- **Transformer Encoder**: Multi-head self-attention mechanism with 4 attention heads to capture inter-feature relationships.
- **Feed-Forward Networks**: Dense layers with 128 units for feature transformation.
- **Output Projection**: Dense layer predicting the next 5 time steps (5 minutes) for all 7 metrics.

Training was performed using the Adam optimizer with default hyperparameters, minimizing mean squared error (MSE) loss across all predicted features. The dataset was scaled using MinMaxScaler to ensure feature values ranged between 0 and 1, with special handling for percentage-based metrics (CPU and memory usage) to maintain realistic bounds.

The training process involved 30 epochs with a batch size of 64, utilizing TensorFlow/Keras for implementation. Model weights and scaler parameters were saved for deployment in the live monitoring system. This approach enabled accurate forecasting of resource utilization patterns, particularly for detecting impending CPU overload conditions.

## 6.1.4 Evaluation Metrics

Model performance for multi-metric time-series prediction was assessed using regression metrics tailored to the forecasting task, focusing on both individual feature accuracy and overall prediction reliability.

**Mean Squared Error (MSE)**: Quantified the average squared difference between predicted and actual values across all features and time steps, providing a comprehensive measure of prediction accuracy.

**Mean Absolute Error (MAE)**: Measured the average absolute deviation of predictions from actual values, offering interpretable error magnitudes in the original feature scales.

**Root Mean Squared Error (RMSE)**: Provided error measurements in the same units as the original metrics, facilitating comparison with real-world thresholds (e.g., CPU usage percentages).

For anomaly detection and remediation evaluation, additional metrics were employed:

- **Detection Accuracy**: Proportion of correctly identified risk conditions (true positives) versus false alarms.
- **Remediation Success Rate**: Percentage of automated actions that successfully mitigated detected issues without causing system instability.
- **Cooldown Effectiveness**: Assessment of scaling cooldown periods to prevent over-reaction to transient spikes.

Performance was evaluated on both historical validation data and live monitoring scenarios, with particular emphasis on CPU usage predictions due to their critical role in triggering remediation actions. The combined use of these metrics ensured a holistic evaluation of the system's ability to predict and respond to Kubernetes workload dynamics.

## 6.2 Coding Details and Code Efficiency

The project code focuses on efficient implementation of deep learning-based predictive monitoring and automated remediation for Kubernetes environments.

### Data Preprocessing and Model Training

```python
def load_data():
    df = pd.read_csv(FILE_PATH)
    df = df[FEATURES].dropna()
    df = df.tail(2000)  # use last 2000 rows for training
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    # Ensure CPU/memory percentages are 0–100
    scaler.data_min_[4] = 0
    scaler.data_max_[4] = 100
    scaler.data_min_[5] = 0
    scaler.data_max_[5] = 100
    joblib.dump(scaler, "multi_scaler.pkl")
    return scaled
```

The preprocessing pipeline loads historical CSV data, handles missing values through dropping, and applies MinMax scaling with bounds checking for percentage metrics. This ensures numerical stability and realistic predictions.

### Sequence Creation for Time-Series Forecasting

```python
def create_sequences(data: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    n, n_feat = data.shape
    for end in range(seq_len, n - horizon):
        start = end - seq_len
        X.append(data[start:end])
        y.append(data[end : end + horizon].reshape(-1))
    return np.array(X), np.array(y)
```

Vectorized operations create sliding windows of historical data, generating input-output pairs for supervised learning. This approach efficiently handles large datasets without explicit loops.

### Transformer Model Architecture

```python
def build_model(seq_len: int, n_feat: int, horizon: int):
    inputs = layers.Input(shape=(seq_len, n_feat))
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=n_feat)(inputs, inputs)
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
```

The model implements a simplified Transformer encoder with residual connections and layer normalization, optimized for multi-variate time-series prediction. Global average pooling reduces computational complexity while preserving temporal information.

### Live Monitoring and Prometheus Integration

```python
def fetch_features_for_workload(namespace: str, deployment: str) -> np.ndarray:
    cpu_usage = prom_query_instant(
        f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{deployment}-.*"}}[1m]))'
    )
    # ... additional metric queries
    return np.array([cpu_eff, mem_eff, disk_io, net_latency, cpu_pct, mem_pct, node_temp], dtype=np.float32)
```

Real-time metric collection uses Prometheus instant queries, with error handling and fallbacks to ensure robust data acquisition. Vectorized NumPy operations maintain efficiency in the monitoring loop.

### Automated Remediation Logic

```python
if stress_detected:
    last_scale = last_scale_ts.get(key, 0)
    if now_ts - last_scale > SCALE_COOLDOWN_SEC:
        last_scale_ts[key] = now_ts
        log_remediation(f"{key}: CPU stress detected. Triggering remediation.")
        simple_remediation("node_cpu_usage", np.array([0]))
```

Cooldown mechanisms prevent over-scaling, with logging for auditability. The remediation engine integrates with Kubernetes API for deployment scaling and pod management.

Throughout the codebase, optimized libraries (NumPy, TensorFlow, scikit-learn) and vectorized computations ensure scalability. Asynchronous monitoring loops and efficient data structures (deques for history) minimize memory usage and computational overhead, enabling real-time operation on resource-constrained environments.
