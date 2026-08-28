# Week 11 — Kubernetes, Helm, Monitoring and the Dashboard

> **DoD** — `helm install` brings up every service on k3d; Grafana shows live
> dashboards; the UI exposes every feature without a terminal.

**Verdict: MET.** A k3d cluster was created, the five service images imported,
and `helm install` brought up all eight workloads. Prometheus scrapes six
targets, Grafana is live, and a Next.js dashboard exposes every service without
touching a terminal.

## helm install on k3d

```
k3d cluster create smartflow --agents 1 \
    --volume /path/to/smartflow:/mnt/smartflow@all \
    --port "30001-30005:30001-30005@server:0"
k3d image import smartflow/{sim,rl,vision,graph,llm}_service:latest -c smartflow
helm install smartflow deploy/helm/smartflow --wait
```

Result — all eight pods Running:

| Pod | Node |
|---|---|
| `sim-service` | agent-0 |
| `rl-service` | agent-0 |
| `vision-service` | agent-0 |
| `llm-service` | agent-0 |
| `graph-service` | server-0 |
| `redpanda` | server-0 |
| `prometheus` | server-0 |
| `grafana` | server-0 |

Verified through the cluster's NodePorts, not just by pod status: the deployed
services return the committed benchmark numbers (`fixed` 83.10 s, `actuated`
40.89 s), the graph service returns `C2 feeds into B2, C1, C3, D2` with 4
incoming lanes and a 4-phase 90 s program, the analytics service reaches the
graph service over cluster DNS, and an unauthenticated request returns **401**.

Prometheus on the cluster reports **6 of 6 targets up**; Grafana answers
`{"database": "ok", "version": "11.5.1"}`.

## The chart, and why the manifests are generated

The chart describes the five services as **data** in `values.yaml` and renders
them with one template, rather than as five near-identical files that drift apart
the first time one is edited. `helm lint` passes; `helm template` renders exactly
18 objects.

`deploy/k8s/smartflow.yaml` holds the same deployment as plain manifests, for
anyone applying it without Helm. It is **generated** by
`deploy/render_manifests.py` from the same `values.yaml`, so the two cannot
disagree — and CI fails if the committed copy is stale. Both paths independently
produce 18 objects, which is a useful cross-check that the generator and the
chart agree.

The generator also validates offline. `kubectl apply --dry-run=client` needs a
live API server for group discovery and is useless on a machine with no cluster,
so `check_structure()` verifies what actually goes wrong when hand-editing
values: a Service selecting an app with no Deployment, a probe pointing at a port
name nothing declares, a duplicate or out-of-range nodePort, an image with no tag.

### The hostPath decision

The data volume is a `hostPath` mount of the repository. That is the right call
for a single-node k3d cluster and the wrong one for a real cluster, so it is
stated plainly rather than presented as production practice: a real deployment
would use a PVC or object storage. It is what makes `helm install` work without
first pushing gigabytes of metrics into a volume.

## The dashboard

`dashboard/` is a Next.js 15 app exposing every feature the DoD lists:

| Panel | Service | What it shows |
|---|---|---|
| Services | all five | Live health and auth mode |
| Benchmark results | `sim` | Per-controller means, with a scenario switcher (base, peak, light, asymmetric) |
| Perception | `vision` | Detector mAP50/recall **with its caveat**, and incident-detection recall, precision and latency |
| Federated learning | `rl` | The Week 9 result, including that its DoD was **not** met and why |
| Knowledge graph | `graph` | All 16 junctions; click one to see what it feeds, its lanes, sensors and phases |
| Ask the corridor | `llm` | Grounded question answering, with refusals shown as refusals |
| Footer | — | Grafana, Prometheus and both OpenAPI docs pages |

**No number on the page is hard-coded.** Every panel fetches from a service,
which reads the committed artifacts. A stale panel therefore means a stale
service, not a stale page.

The browser never holds a bearer token: it calls `/api/proxy`, a server-side
route that mints the token and calls the service over the internal network. That
also removes the need to configure CORS on five separate services. The proxy
accepts only GET, plus POST to exactly one path, so it cannot be turned into an
open relay onto the cluster network.

## Two defects found by building the UI

Both were invisible until something rendered the data:

- **The graph service reported zero incoming lanes for every junction.** Lanes
  hang off roads (`Road -HAS_LANE-> Lane`), and roads off junctions
  (`Road -ENDS_AT-> Junction`); walking only a junction's own outbound edges
  finds sensors and the program but no lanes. The dashboard showed "0 lanes"
  next to "4 sensors", which is obviously wrong in a way a JSON response is not.
- **Signal programs reported zero phases** for the same reason: phases are their
  own nodes linked by `HAS_PHASE`. Both now traverse correctly — C2 returns 4
  lanes, 4 sensors and a 4-phase 90 s cycle on both deployments.

## Running it

```
docker compose up -d --build          # or the k3d path above
cd dashboard && npm install && npm run dev
```

Dashboard on `localhost:3000`, Grafana on `localhost:3001` (Compose) or
`localhost:30300` (k3d), Prometheus on `9090` / `30090`.
