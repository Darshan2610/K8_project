# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import MinMaxScaler
# # import joblib
# # import tensorflow as tf
# # from tensorflow.keras import layers, models

# # # Load dataset
# # file_path = "kubernetes_performance_metrics_dataset.csv"
# # df = pd.read_csv(file_path)

# # # Select relevant features
# # features = [
# #     "cpu_allocation_efficiency",
# #     "memory_allocation_efficiency",
# #     "disk_io",
# #     "network_latency",
# #     "node_cpu_usage",
# #     "node_memory_usage",
# #     "node_temperature",
# # ]

# # df = df[features].dropna()
# # # Use a subset for faster training
# # df = df.head(500)

# # # Scale features
# # scaler = MinMaxScaler()
# # scaled_data = scaler.fit_transform(df)
# # joblib.dump(scaler, "scaler_k8s.pkl")

# # seq_len = 20
# # X, y = [], []
# # for i in range(seq_len, len(scaled_data)):
# #     X.append(scaled_data[i-seq_len:i])
# #     y.append(scaled_data[i])
# # X = np.array(X)
# # y = np.array(y)

# # # Build a simple transformer model
# # inputs = layers.Input(shape=(seq_len, X.shape[2]))
# # attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=X.shape[2])(inputs, inputs)
# # attn_output = layers.Add()([inputs, attn_output])
# # attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)
# # ffn = layers.Dense(64, activation="relu")(attn_output)
# # ffn = layers.Dense(X.shape[2])(ffn)
# # ffn_output = layers.Add()([attn_output, ffn])
# # ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output)
# # pooled = layers.GlobalAveragePooling1D()(ffn_output)
# # outputs = layers.Dense(X.shape[2])(pooled)
# # model = models.Model(inputs=inputs, outputs=outputs)

# # model.compile(optimizer="adam", loss="mse")
# # model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# # model.save("transformer_k8s_model.h5")

















# #CODEX CODE FOR PLOTS







# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# # Load dataset
# file_path = "kubernetes_performance_metrics_dataset.csv"
# df = pd.read_csv(file_path)

# # Select relevant features
# features = [
#     "cpu_allocation_efficiency",
#     "memory_allocation_efficiency",
#     "disk_io",
#     "network_latency",
#     "node_cpu_usage",
#     "node_memory_usage",
#     "node_temperature",
# ]

# df = df[features].dropna()

# # Use a subset for faster training
# df = df.head(500)

# # Scale features
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)
# joblib.dump(scaler, "scaler_k8s.pkl")

# seq_len = 20
# X, y = [], []
# for i in range(seq_len, len(scaled_data)):
#     X.append(scaled_data[i - seq_len : i])
#     y.append(scaled_data[i])
# X = np.array(X)
# y = np.array(y)

# # Build a simple transformer model
# inputs = layers.Input(shape=(seq_len, X.shape[2]))
# attn_output = layers.MultiHeadAttention(
#     num_heads=2,
#     key_dim=X.shape[2],          # attention key dimension
#     output_shape=X.shape[2]      # ensures same final dim for residual add
# )(inputs, inputs)
# attn_output = layers.Add()([inputs, attn_output])
# attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)

# ffn = layers.Dense(64, activation="relu")(attn_output)
# ffn = layers.Dense(X.shape[2])(ffn)
# ffn_output = layers.Add()([attn_output, ffn])
# ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output)

# pooled = layers.GlobalAveragePooling1D()(ffn_output)
# outputs = layers.Dense(X.shape[2])(pooled)
# model = models.Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer="adam", loss="mse")
# model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# model.save("transformer_k8s_model.h5")

# # Generate predictions and plot against actual values
# predictions = model.predict(X, verbose=0)
# actual = scaler.inverse_transform(y)
# predicted = scaler.inverse_transform(predictions)

# plt.figure(figsize=(10, 14))
# for i, feature in enumerate(features):
#     plt.subplot(len(features), 1, i + 1)
#     plt.plot(actual[:, i], label="Actual")
#     plt.plot(predicted[:, i], label="Predicted")
#     plt.title(feature)
#     plt.xlabel("Sample")
#     plt.ylabel("Value")
#     plt.legend()

# plt.tight_layout()
# plt.savefig("prediction_plot.png")
# plt.close()











#MODIFIED CODE TO DISPLAY ANOMALIES






import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


FILE_PATH = "kubernetes_performance_metrics_dataset.csv"
SEQ_LEN = 20
MAD_K = 3.0          # sensitivity for MAD (higher = fewer anomalies)
FALLBACK_Q = 0.98    # fallback percentile if MAD finds none
MIN_PTS = 5          


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

df = df[features].dropna().head(500)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
joblib.dump(scaler, "scaler_k8s.pkl")

X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])
    y.append(scaled[i])
X = np.array(X)
y = np.array(y)

#model
inputs = layers.Input(shape=(SEQ_LEN, X.shape[2]))
attn = layers.MultiHeadAttention(num_heads=2, key_dim=X.shape[2])(inputs, inputs)
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
model.fit(X, y, epochs=20, batch_size=32, verbose=0)
model.save("transformer_k8s_model.h5")

#prediction
pred_scaled = model.predict(X, verbose=0)
actual = scaler.inverse_transform(y)
predicted = scaler.inverse_transform(pred_scaled)

#detection
def mad_based_threshold(res):
    """Return anomaly indices using robust MAD threshold; fall back if none."""
    # res: 1D residuals (absolute)
    med = np.median(res)
    mad = np.median(np.abs(res - med))
    # avoid zero MAD (flat series)
    robust_std = 1.4826 * mad if mad > 0 else np.std(res) + 1e-8
    thr = med + MAD_K * robust_std
    idx = np.where(res > thr)[0]

    if idx.size == 0:
        # Fallback: top residuals by percentile, ensure a few points
        qthr = np.quantile(res, FALLBACK_Q)
        idx = np.where(res >= qthr)[0]
        if idx.size < MIN_PTS:
            idx = np.argsort(res)[-MIN_PTS:]
    return np.sort(idx)

residuals = np.abs(actual - predicted)  # (N, F)
anom_idx_by_feat = [mad_based_threshold(residuals[:, i]) for i in range(len(features))]

#ploting
plt.figure(figsize=(10, 14))
for i, feat in enumerate(features):
    plt.subplot(len(features), 1, i + 1)
    plt.plot(actual[:, i], label="Actual")
    plt.plot(predicted[:, i], label="Predicted")

    ai = anom_idx_by_feat[i]
    plt.scatter(ai, actual[ai, i], s=28, color="red", marker="o",
                edgecolors="black", linewidths=0.4, zorder=6, label="Anomaly")

    plt.title(feat)
    plt.xlabel("Sample")
    plt.ylabel("Value")

    # De-duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    plt.legend(H, L, loc="best")

plt.tight_layout()
plt.savefig("prediction_plot_with_anomalies.png")
plt.close()

print("Saved: prediction_plot_with_anomalies.png")

