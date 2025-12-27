# collector.py
import requests
import csv
import time
from datetime import datetime

PROM_URL = "http://localhost:9090"

FEATURES = [
    "cpu_allocation_efficiency",
    "memory_allocation_efficiency",
    "disk_io",
    "network_latency",
    "node_cpu_usage",
    "node_memory_usage",
    "node_temperature",
]

CSV_FILE = "data/live_metrics.csv"


def prom(expr: str):
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": expr}, timeout=5)
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            return 0.0
        return float(res[0]["value"][1])
    except:
        return 0.0


def collect_once():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # --- Core usage ---
    cpu_raw = prom(
        'sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"nginx-.*",image!=""}[5m]))'
    )
    cpu_cores = cpu_raw / 1e9

    # --- Memory usage ---
    mem_bytes = prom(
        'sum(container_memory_working_set_bytes{namespace="default",pod=~"nginx-.*",image!=""})'
    )

    # --- Requests ---
    cpu_req = (
        prom(
            'sum(kube_pod_container_resource_requests_cpu_cores{namespace="default",pod=~"nginx-.*"})'
        )
        or 1e-6
    )
    mem_req = (
        prom(
            'sum(kube_pod_container_resource_requests_memory_bytes{namespace="default",pod=~"nginx-.*"})'
        )
        or 1e-6
    )

    cpu_eff = cpu_cores / cpu_req
    mem_eff = mem_bytes / mem_req

    # --- Disk I/O ---
    disk_io = prom(
        "sum(rate(node_disk_read_bytes_total[5m]) + rate(node_disk_written_bytes_total[5m]))"
    )

    # --- Latency ---
    network_latency = prom(
        "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
    )

    # --- CPU % ---
    total_cpu = prom("sum(machine_cpu_cores)") or 1
    cpu_pct = (cpu_cores / total_cpu) * 100

    # --- Memory % ---
    total_mem = prom("sum(machine_memory_bytes)") or 1
    mem_pct = (mem_bytes / total_mem) * 100

    # --- Temperature ---
    temp_raw = prom("avg(node_thermal_zone_temp)")
    temp = temp_raw * 0.001 if temp_raw > 100 else temp_raw

    row = [
        timestamp,
        cpu_eff,
        mem_eff,
        disk_io,
        network_latency,
        cpu_pct,
        mem_pct,
        temp,
    ]

    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

    print("Collected row:", row)


if __name__ == "__main__":
    print("Collector running every 60s…")
    while True:
        collect_once()
        time.sleep(60)
