# stress_pod.py
import subprocess

NAMESPACE = "default"
LABEL_SELECTOR = "app=nginx"
STRESS_SECONDS = 30  # duration of stress


def kubectl_get_pod_by_label(namespace: str, label_selector: str) -> str:
    cmd = [
        "kubectl",
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        label_selector,
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    if not out:
        raise RuntimeError(f"No pod found for label {label_selector} in {namespace}")
    return out


def start_cpu_mem_stress(pod_name: str, namespace: str, seconds: int):
    # Light CPU + memory stress using shell only
    # ~8 MB memory allocation, light CPU loop
    stress_script = (
        'echo "[pod] Starting light stress"; '
        'awk "BEGIN { '
        "  a[0]=0; for(i=0;i<1024*1024;i++) a[i]=i; "  # ~8 MB memory
        f"  end=systime()+{seconds}; x=0; while(systime()<end) x=x+1; "
        '  print \\"[pod] Light stress finished\\"; '
        '}"'
    )

    cmd = [
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "--",
        "sh",
        "-c",
        stress_script,
    ]
    print(
        f"Starting light CPU+memory stress in {namespace}/{pod_name} for {seconds}s ..."
    )
    subprocess.call(cmd)
    print("Stress finished.")


if __name__ == "__main__":
    try:
        pod = kubectl_get_pod_by_label(NAMESPACE, LABEL_SELECTOR)
        print(f"Using pod: {NAMESPACE}/{pod}")
        start_cpu_mem_stress(pod, NAMESPACE, STRESS_SECONDS)
    except Exception as e:
        print(f"Failed to start stress: {e}")
