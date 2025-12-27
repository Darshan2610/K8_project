# stress_pod.py
import subprocess

NAMESPACE = "default"
LABEL_SELECTOR = "app=nginx"
STRESS_SECONDS = 40   # shorter duration, safe default

def kubectl_get_pod_by_label(namespace: str, label_selector: str) -> str:
    cmd = [
        "kubectl", "get", "pods", "-n", namespace,
        "-l", label_selector, "-o", "jsonpath={.items[0].metadata.name}",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    if not out:
        raise RuntimeError(f"No pod found for label {label_selector} in {namespace}")
    return out

def start_cpu_mem_stress(pod_name: str, namespace: str, seconds: int):
    # Allocate ~10 MB RAM and run a simple loop for CPU
    pod_script = (
        'echo "Starting safe stress"; '
        'awk "BEGIN { '
        '  print \\"[pod] allocating memory...\\"; '
        '  for(i=0; i<1024*1024*2; i++) a[i]=i; '   # ~8 MB RAM
        '  print \\"[pod] memory allocated\\"; '
        f'  end = systime() + {seconds}; '
        '  while (systime() < end) { x = x + 1; } '
        '  print \\"[pod] Stress finished\\"; '
        '}"'
    )

    cmd = [
        "kubectl", "exec", "-n", namespace, pod_name,
        "--", "sh", "-c", pod_script
    ]

    print(f"Starting safe CPU+memory stress in {namespace}/{pod_name} for {seconds}s ...")
    subprocess.call(cmd)
    print("Stress finished.")

if __name__ == "__main__":
    try:
        pod = kubectl_get_pod_by_label(NAMESPACE, LABEL_SELECTOR)
        print(f"Using pod: {NAMESPACE}/{pod}")
        start_cpu_mem_stress(pod, NAMESPACE, STRESS_SECONDS)
    except Exception as e:
        print(f"Failed to start stress: {e}")
