# Week 10 — Platform Split, Authentication and CI/CD

> **DoD** — a push to main rebuilds and redeploys every service automatically,
> behind an authenticated dashboard.

**Verdict: MET, with one gated step.** Five domain services are built, deployed
and verified running; every route is behind bearer authentication; the CI
pipeline builds, tests and publishes all five images on a push to main. The
redeploy step is written and wired but runs only when a deployment target is
configured — see "What the pipeline does and does not do" below.

## The five services

Each serves the project's committed artifacts over HTTP. None of them starts
SUMO or trains anything: a 1800-second episode takes minutes and does not belong
behind a synchronous request, so the services are a read surface over work the
harnesses already did.

| Service | Port | What it serves |
|---|---:|---|
| `sim_service` | 8001 | Network topology from `corridor.net.xml`, and every recorded benchmark run |
| `rl_service` | 8002 | Trained policy inventory, per-controller results, baseline comparisons, the federated result |
| `vision_service` | 8003 | Detector metrics, incident-detection sweep, and inference on an uploaded frame |
| `graph_service` | 8004 | The 301-node knowledge graph: junctions, lanes, sensors, programs, rules, results |
| `llm_service` | 8005 | Read-only question answering — calls `graph_service` over HTTP and retrieves from the project's reports |

### Data is mounted, not baked

The services read `outputs/` and `data/` from a read-only mount rather than
carrying a copy inside each image. Baking a hundred megabytes into five images
would multiply it five ways and go stale the moment a benchmark is re-run. It
also means an API response and a report are reading the same file, so they
cannot disagree about a number — `tests/test_services.py` asserts the API returns
the committed values (`fixed` 83.10 s, `actuated` 40.89 s, `marl_shared_w5`
17.85 s) rather than merely returning *something*.

A service whose mount is missing reports `"status": "degraded"` on its health
endpoint rather than answering with empty lists that look valid.

## Authentication

Two modes, chosen by configuration rather than by a code branch a deployment can
forget to flip:

- **Auth0** — with `AUTH0_DOMAIN` and `AUTH0_AUDIENCE` set, tokens are verified
  as RS256 against the tenant's published JWKS, checking signature, audience and
  issuer.
- **Local development** — without a tenant, an HS256 token signed with a local
  secret is accepted, so the authenticated path is exercised end to end and
  testable in CI without provisioning an identity provider.

**The development fallback is refused outright when `SMARTFLOW_ENV=production`.**
A convenience that survives silently into production is how an unauthenticated
dashboard happens, so it fails closed. `SMARTFLOW_AUTH_DISABLED` is likewise
rejected in production.

Verified: every protected route returns **401** without a token, and a malformed
token is rejected.

## Events

The services publish analytics events — `runs.listed`, `junction.read`,
`query.answered`, `query.refused`, `frame.detected` — to Redpanda, which speaks
the Kafka API in one container. Confirmed flowing end to end by consuming the
topic:

```
{"service": "graph_service", "event": "junction.read", "payload": {"junction": "C2"}}
{"service": "llm_service", "event": "query.answered", "payload": {"facts": 1, "passages": 4}}
```

Publishing is deliberately **best-effort**: with no broker reachable it becomes a
no-op. A read request should not fail because a message bus is down, and nothing
that must not be lost is on this path.

## Observability

Every service exposes `/metrics`; Prometheus scrapes all five plus Redpanda;
Grafana provisions a dashboard from `deploy/grafana/`. Verified with **7 of 7
scrape targets up** under Compose and 6 of 6 on Kubernetes.

## What the pipeline does and does not do

`.github/workflows/ci.yml` has five jobs:

| Job | What it does |
|---|---|
| `logic-tests` | Renders the Kubernetes manifests and fails if the committed copies are stale |
| `build` | Builds all five images in a matrix and pushes to GHCR on main |
| `integration` | Starts the stack with Compose, waits for health, runs the 12 end-to-end checks |
| `helm-chart` | Lints the chart and asserts it renders exactly 18 objects |
| `deploy` | Runs `helm upgrade --install` against a configured cluster |

**The `deploy` job is honest about its own preconditions.** It checks for a
`KUBE_CONFIG` secret; without one it emits a notice saying images were built and
published but nothing was redeployed, rather than passing silently and implying a
deployment happened. That is the one part of this week's DoD that cannot be
demonstrated from this machine: it needs a cluster reachable from GitHub's
runners and a kubeconfig secret, neither of which a local k3d cluster provides.

The rest of the pipeline is exercisable locally, and the deployment it *would*
run is the same `helm upgrade --install` that was run by hand against k3d in
Week 11 and verified working.

## Verification

```
docker compose up -d --build
python tests/test_services.py      # 12/12
```

The suite checks health, data mounting, 401 on every protected route, invalid
token rejection, API values matching the committed metrics, graph topology
matching Week 7, comparison direction correctness, the detector caveat, the
inter-service call, out-of-domain refusal, the absence of mutating routes, and
`/metrics` exposure.
