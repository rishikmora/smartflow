"""Render plain Kubernetes manifests from the Helm chart's values.

The chart and the raw manifests describe the same deployment, so they are
generated from one source — ``deploy/helm/smartflow/values.yaml`` — rather than
maintained as two files that drift apart the first time one is edited.

Raw manifests exist because ``kubectl apply -f deploy/k8s`` needs no Helm
installed, which matters for anyone reproducing this without the full toolchain.

Usage:
    python deploy/render_manifests.py
    python deploy/render_manifests.py --validate    # also run kubectl dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
VALUES = os.path.join(HERE, "helm", "smartflow", "values.yaml")
OUT_DIR = os.path.join(HERE, "k8s")

LABELS = {"app.kubernetes.io/part-of": "smartflow"}


def load_values() -> dict:
    """Read the chart's values.

    Returns:
        The parsed values document.

    Raises:
        FileNotFoundError: if values.yaml is missing.
        RuntimeError: if PyYAML is unavailable.
    """
    if not os.path.isfile(VALUES):
        raise FileNotFoundError(f"Chart values not found at {VALUES}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with io.open(VALUES, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _labels(app: str) -> dict:
    """Build the standard label set for one object.

    Args:
        app: the app label.

    Returns:
        Labels including the shared part-of label.
    """
    return {"app": app, **LABELS}


def render(values: dict) -> list[dict]:
    """Build every Kubernetes object for the stack.

    Args:
        values: the chart values.

    Returns:
        A list of manifest dictionaries, in apply order.
    """
    objects: list[dict] = []
    registry = values["image"].get("registry") or ""
    tag = values["image"]["tag"]

    def image(repo: str) -> str:
        return f"{registry}/{repo}:{tag}" if registry else f"{repo}:{tag}"

    objects.append({
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {"name": values["auth"]["devJwtSecretName"], "labels": dict(LABELS)},
        "type": "Opaque",
        "stringData": {"devJwtSecret": "smartflow-dev-secret"},
    })

    for service in values["services"]:
        name = service["name"]
        env = [
            {"name": "SMARTFLOW_DATA_ROOT", "value": "/data"},
            {"name": "SMARTFLOW_ENV", "value": str(values["env"]["SMARTFLOW_ENV"])},
            {"name": "SMARTFLOW_EVENT_TOPIC",
             "value": str(values["env"]["SMARTFLOW_EVENT_TOPIC"])},
            {"name": "AUTH0_DOMAIN", "value": str(values["auth"]["auth0Domain"])},
            {"name": "AUTH0_AUDIENCE", "value": str(values["auth"]["auth0Audience"])},
            {"name": "SMARTFLOW_DEV_JWT_SECRET",
             "valueFrom": {"secretKeyRef": {
                 "name": values["auth"]["devJwtSecretName"], "key": "devJwtSecret"}}},
        ]
        if values["redpanda"]["enabled"]:
            env.append({"name": "REDPANDA_BROKERS", "value": "redpanda:9092"})
        for key, value in (service.get("extraEnv") or {}).items():
            env.append({"name": key, "value": str(value)})

        objects.append({
            "apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": name, "labels": _labels(name)},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": _labels(name)},
                    "spec": {
                        "securityContext": {"runAsNonRoot": True, "runAsUser": 10001},
                        "containers": [{
                            "name": name,
                            "image": image(service["image"]),
                            "imagePullPolicy": values["image"]["pullPolicy"],
                            "ports": [{"name": "http",
                                       "containerPort": service["port"]}],
                            "env": env,
                            "volumeMounts": [{"name": "project-data",
                                              "mountPath": "/data",
                                              "readOnly": values["data"]["readOnly"]}],
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": "http"},
                                "initialDelaySeconds": 5, "periodSeconds": 10},
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": "http"},
                                "initialDelaySeconds": 20, "periodSeconds": 20},
                            "resources": values["resources"],
                        }],
                        "volumes": [{"name": "project-data",
                                     "hostPath": {"path": values["data"]["hostPath"],
                                                  "type": "Directory"}}],
                    },
                },
            },
        })
        objects.append({
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": name, "labels": _labels(name)},
            "spec": {"type": "NodePort", "selector": {"app": name},
                     "ports": [{"name": "http", "port": service["port"],
                                "targetPort": "http",
                                "nodePort": service["nodePort"]}]},
        })

    if values["redpanda"]["enabled"]:
        objects.append({
            "apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": {"name": "redpanda", "labels": _labels("redpanda")},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "redpanda"}},
                "template": {
                    "metadata": {"labels": _labels("redpanda")},
                    "spec": {"containers": [{
                        "name": "redpanda",
                        "image": values["redpanda"]["image"],
                        "args": ["redpanda", "start", "--overprovisioned", "--smp=1",
                                 f"--memory={values['redpanda']['memory']}",
                                 "--reserve-memory=0M", "--node-id=0", "--check=false",
                                 "--kafka-addr=PLAINTEXT://0.0.0.0:9092",
                                 "--advertise-kafka-addr=PLAINTEXT://redpanda:9092"],
                        "ports": [{"name": "kafka", "containerPort": 9092},
                                  {"name": "admin", "containerPort": 9644}],
                        "readinessProbe": {"tcpSocket": {"port": "kafka"},
                                           "initialDelaySeconds": 10,
                                           "periodSeconds": 10},
                    }]},
                },
            },
        })
        objects.append({
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": "redpanda", "labels": _labels("redpanda")},
            "spec": {"selector": {"app": "redpanda"},
                     "ports": [{"name": "kafka", "port": 9092, "targetPort": "kafka"},
                               {"name": "admin", "port": 9644, "targetPort": "admin"}]},
        })

    if values["monitoring"]["enabled"]:
        targets = "\n".join(f"              - {s['name']}:{s['port']}"
                            for s in values["services"])
        scrape = ("global:\n  scrape_interval: 15s\n"
                  "scrape_configs:\n"
                  "  - job_name: smartflow-services\n"
                  "    metrics_path: /metrics\n"
                  "    static_configs:\n"
                  "      - targets:\n" + targets + "\n"
                  "        labels:\n          project: smartflow\n")
        if values["redpanda"]["enabled"]:
            scrape += ('  - job_name: redpanda\n    static_configs:\n'
                       '      - targets: ["redpanda:9644"]\n')
        objects.append({
            "apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": "prometheus-config", "labels": dict(LABELS)},
            "data": {"prometheus.yml": scrape},
        })
        for app, spec in (
            ("prometheus", {"image": values["monitoring"]["prometheus"]["image"],
                            "port": 9090,
                            "nodePort": values["monitoring"]["prometheus"]["nodePort"]}),
            ("grafana", {"image": values["monitoring"]["grafana"]["image"],
                         "port": 3000,
                         "nodePort": values["monitoring"]["grafana"]["nodePort"]}),
        ):
            container: dict = {
                "name": app, "image": spec["image"],
                "ports": [{"name": "http", "containerPort": spec["port"]}],
            }
            volumes: list[dict] = []
            if app == "prometheus":
                container["volumeMounts"] = [{"name": "config",
                                              "mountPath": "/etc/prometheus"}]
                volumes = [{"name": "config",
                            "configMap": {"name": "prometheus-config"}}]
            else:
                container["env"] = [
                    {"name": "GF_SECURITY_ADMIN_PASSWORD",
                     "value": str(values["monitoring"]["grafana"]["adminPassword"])},
                    {"name": "GF_AUTH_ANONYMOUS_ENABLED", "value": "true"},
                    {"name": "GF_AUTH_ANONYMOUS_ORG_ROLE", "value": "Viewer"},
                ]
            pod_spec: dict = {"containers": [container]}
            if volumes:
                pod_spec["volumes"] = volumes
            objects.append({
                "apiVersion": "apps/v1", "kind": "Deployment",
                "metadata": {"name": app, "labels": _labels(app)},
                "spec": {"replicas": 1,
                         "selector": {"matchLabels": {"app": app}},
                         "template": {"metadata": {"labels": _labels(app)},
                                      "spec": pod_spec}},
            })
            objects.append({
                "apiVersion": "v1", "kind": "Service",
                "metadata": {"name": app, "labels": _labels(app)},
                "spec": {"type": "NodePort", "selector": {"app": app},
                         "ports": [{"name": "http", "port": spec["port"],
                                    "targetPort": "http",
                                    "nodePort": spec["nodePort"]}]},
            })

    return objects


def write(objects: list[dict]) -> str:
    """Write the manifests as one applyable YAML file.

    Args:
        objects: rendered manifest dictionaries.

    Returns:
        Path to the written file.
    """
    import yaml

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "smartflow.yaml")
    header = (
        "# SmartFlow - Kubernetes manifests.\n"
        "#\n"
        "# GENERATED by deploy/render_manifests.py from deploy/helm/smartflow/values.yaml.\n"
        "# Edit the values, not this file. The Helm chart and these manifests describe\n"
        "# the same deployment; this exists so the stack can be applied without Helm.\n"
        "#\n"
        "#   kubectl apply -f deploy/k8s/smartflow.yaml\n"
        "#\n"
        "# The data volume is a hostPath: on a single-node k3d cluster, mount the\n"
        "# repository at values.data.hostPath, e.g.\n"
        "#   k3d cluster create smartflow --volume /path/to/smartflow:/mnt/smartflow\n"
    )
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        for obj in objects:
            handle.write("---\n")
            yaml.safe_dump(obj, handle, sort_keys=False, default_flow_style=False)
    return path


def check_structure(objects: list[dict]) -> list[str]:
    """Validate the manifests offline, without a cluster.

    ``kubectl apply --dry-run=client`` needs a live API server for group
    discovery, so it cannot check anything on a machine with no cluster. These
    are the checks that catch the mistakes that actually happen when hand-editing
    values: a Service with no matching Deployment, a probe pointing at a port
    name nothing declares, a duplicate nodePort, a missing image tag.

    Args:
        objects: rendered manifest dictionaries.

    Returns:
        A list of problems; empty means the manifests are internally consistent.
    """
    problems: list[str] = []
    deployments = {o["metadata"]["name"]: o for o in objects if o["kind"] == "Deployment"}
    services = [o for o in objects if o["kind"] == "Service"]
    node_ports: dict[int, str] = {}

    for obj in objects:
        for field in ("apiVersion", "kind", "metadata"):
            if field not in obj:
                problems.append(f"{obj.get('kind', '?')}: missing {field}")
        if not obj.get("metadata", {}).get("name"):
            problems.append(f"{obj.get('kind', '?')}: missing metadata.name")

    for name, deployment in deployments.items():
        pod = deployment["spec"]["template"]["spec"]
        selector = deployment["spec"]["selector"]["matchLabels"]
        labels = deployment["spec"]["template"]["metadata"]["labels"]
        if not all(labels.get(k) == v for k, v in selector.items()):
            problems.append(f"{name}: selector does not match pod labels")
        for container in pod["containers"]:
            if ":" not in container.get("image", ""):
                problems.append(f"{name}: image {container.get('image')!r} has no tag")
            declared = {p.get("name") for p in container.get("ports", [])}
            for probe in ("readinessProbe", "livenessProbe"):
                target = (container.get(probe) or {}).get("httpGet", {}).get("port")
                if target and target not in declared:
                    problems.append(
                        f"{name}: {probe} targets port {target!r}, "
                        f"which the container does not declare {sorted(declared)}")

    for service in services:
        name = service["metadata"]["name"]
        app = service["spec"]["selector"].get("app")
        if app and app not in deployments:
            problems.append(f"Service {name}: selects app={app!r} with no Deployment")
        for port in service["spec"]["ports"]:
            node_port = port.get("nodePort")
            if node_port is None:
                continue
            if not (30000 <= node_port <= 32767):
                problems.append(f"Service {name}: nodePort {node_port} outside 30000-32767")
            if node_port in node_ports:
                problems.append(f"Service {name}: nodePort {node_port} already used by "
                                f"{node_ports[node_port]}")
            node_ports[node_port] = name

    return problems


def validate(path: str) -> bool:
    """Validate the manifests with kubectl's client-side dry run.

    Args:
        path: the manifest file.

    Returns:
        True if kubectl accepted every object.
    """
    try:
        result = subprocess.run(
            ["kubectl", "apply", "--dry-run=client", "--validate=false", "-f", path],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("kubectl unavailable: %s", exc)
        return False
    if result.returncode != 0:
        log.error("kubectl rejected the manifests:\n%s", result.stderr.strip()[:2000])
        return False
    created = [line for line in result.stdout.splitlines() if line.strip()]
    log.info("kubectl validated %d objects", len(created))
    for line in created:
        log.info("  %s", line)
    return True


def main() -> None:
    """Render, write and optionally validate the manifests."""
    parser = argparse.ArgumentParser(description="Render Kubernetes manifests.")
    parser.add_argument("--validate", action="store_true",
                        help="Run kubectl apply --dry-run=client afterwards.")
    args = parser.parse_args()

    values = load_values()
    objects = render(values)

    problems = check_structure(objects)
    if problems:
        for problem in problems:
            log.error("  %s", problem)
        raise SystemExit(f"{len(problems)} structural problems in the manifests")
    log.info("Structural checks passed for %d objects", len(objects))

    path = write(objects)
    kinds: dict[str, int] = {}
    for obj in objects:
        kinds[obj["kind"]] = kinds.get(obj["kind"], 0) + 1
    log.info("Wrote %s (%d objects: %s)", path, len(objects),
             ", ".join(f"{v}x{k}" for k, v in sorted(kinds.items())))

    if args.validate and not validate(path):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
