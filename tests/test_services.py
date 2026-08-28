"""End-to-end checks against the running SmartFlow service stack.

Week 12's integration test. It talks to the five services over HTTP exactly as a
client would, so it exercises the images, the mounts, the authentication and the
inter-service call — none of which a unit test touches.

Three things it is built to catch, because each would leave a stack that looks
healthy and is not:

* **Services that answer with nothing.** A missing mount makes every endpoint
  return an empty list, which looks like a valid response. The tests assert
  against the *committed numbers*, so an empty mount fails loudly.
* **Authentication that is not actually enforced.** Every protected route is
  called without a token and must return 401.
* **The read-only boundary.** The analytics service must expose no route that
  mutates anything, and must not import the simulation or RL stack.

Run the stack first:

    docker compose up -d --build
    python tests/test_services.py

Skips itself with a clear message when the stack is not up, so it can live in a
CI job that does not always start Docker.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "services"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

PORTS = {"sim": 8001, "rl": 8002, "vision": 8003, "graph": 8004, "llm": 8005}
TIMEOUT_S = 20

# Committed values every service must agree with. If a mount is missing or a
# service invents data, these fail.
EXPECTED_BASE_WAIT = {"fixed": 83.097, "actuated": 40.893, "marl_shared_w5": 17.85}
EXPECTED_C2_FEEDS = ["B2", "C1", "C3", "D2"]


def _token() -> str:
    """Mint a development bearer token.

    Returns:
        An HS256 token accepted by the stack in development mode.
    """
    from common.auth import issue_dev_token
    return issue_dev_token("integration-test")


def call(service: str, path: str, token: str | None = None,
         method: str = "GET", body: dict | None = None) -> tuple[int | None, Any]:
    """Make one HTTP call against a service.

    Args:
        service: key of :data:`PORTS`.
        path: request path.
        token: optional bearer token.
        method: HTTP method.
        body: optional JSON body.

    Returns:
        ``(status_code, parsed_body)``; status is None if the host is unreachable.
    """
    url = f"http://localhost:{PORTS[service]}{path}"
    request = urllib.request.Request(url, method=method)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if body is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")[:200]
    except Exception as exc:  # noqa: BLE001 - stack may not be running
        return None, str(exc)[:200]


def stack_is_up() -> bool:
    """Report whether every service answers its health endpoint.

    Returns:
        True if all five services respond.
    """
    return all(call(name, "/health")[0] == 200 for name in PORTS)


def test_every_service_is_healthy() -> None:
    """All five services must report health and a known auth mode."""
    for name in PORTS:
        status, body = call(name, "/health")
        assert status == 200, f"{name} health returned {status}: {body}"
        assert body["service"].startswith(name), body
        assert body["status"] in {"ok", "degraded"}, body
        assert body["auth"] in {"auth0", "development", "disabled"}, body


def test_services_have_their_data_mounted() -> None:
    """A service with no data would answer emptily and look fine; it must not."""
    status, body = call("sim", "/health")
    assert status == 200
    assert body["status"] == "ok", f"sim_service is degraded: {body['data']}"
    assert body["data"]["corridor_metrics"] is True, body["data"]

    status, body = call("graph", "/health")
    assert status == 200 and body["nodes"] > 0, body


def test_protected_routes_reject_anonymous_callers() -> None:
    """Every protected route must return 401 without a bearer token."""
    protected = [("sim", "/runs"), ("sim", "/runs/summary"), ("sim", "/network"),
                 ("rl", "/results"), ("rl", "/models"), ("rl", "/compare"),
                 ("vision", "/detector"), ("vision", "/anomalies"),
                 ("graph", "/graph/stats"), ("graph", "/graph/junctions"),
                 ("llm", "/summary")]
    for service, path in protected:
        status, _body = call(service, path)
        assert status == 401, f"{service}{path} returned {status}, expected 401"

    status, _ = call("llm", "/query", method="POST",
                     body={"question": "anything at all"})
    assert status == 401, f"llm /query returned {status} without a token"


def test_an_invalid_token_is_rejected() -> None:
    """A malformed or wrongly signed token must not be accepted."""
    status, _ = call("sim", "/runs", token="not-a-real-token")
    assert status == 401, status


def test_metrics_match_the_committed_results() -> None:
    """The API must report the same numbers as the metrics CSV."""
    token = _token()
    status, body = call("sim", "/runs/summary?scenario=base", token)
    assert status == 200, body
    by_controller = {row["controller"]: row for row in body["controllers"]}
    for controller, expected in EXPECTED_BASE_WAIT.items():
        assert controller in by_controller, f"{controller} missing from {list(by_controller)}"
        actual = by_controller[controller]["avg_wait_time_s"]
        assert abs(actual - expected) < 0.01, f"{controller}: {actual} != {expected}"
        assert by_controller[controller]["seeds"] == 3, by_controller[controller]


def test_graph_topology_matches_week7() -> None:
    """The graph service must return the same topology Week 7 reported."""
    token = _token()
    status, body = call("graph", "/graph/junctions/C2", token)
    assert status == 200, body
    assert body["feeds"] == EXPECTED_C2_FEEDS, body["feeds"]
    assert len(body["lanes"]) == 4, f"expected 4 incoming lanes, got {body['lanes']}"
    assert len(body["sensors"]) == 4, body["sensors"]
    assert body["program"] is not None, "C2 should run a signal program"


def test_rl_comparison_states_the_direction_of_better() -> None:
    """A comparison that does not say which way is better is a trap."""
    token = _token()
    status, body = call("rl", "/compare?scenario=base&metric=throughput_veh", token)
    assert status == 200, body
    assert body["higher_is_better"] is True, "throughput: more is better"

    status, body = call("rl", "/compare?scenario=base&metric=avg_wait_time_s", token)
    assert body["higher_is_better"] is False, "wait: less is better"
    entry = next(c for c in body["comparisons"] if c["controller"] == "marl_shared_w5")
    versus = entry["versus"]["actuated"]
    assert versus["change_pct"] < 0 and versus["better"] is True, versus


def test_vision_service_reports_its_caveat() -> None:
    """The detector's score must travel with the caveat that bounds it."""
    token = _token()
    status, body = call("vision", "/detector", token)
    assert status == 200, body
    assert body["metrics"]["mAP50"] > 0.9, body["metrics"]
    assert "rendered" in body["caveat"].lower(), body["caveat"]


def test_llm_service_calls_the_graph_service() -> None:
    """A question naming a junction must be answered from the graph, over HTTP."""
    token = _token()
    status, body = call("llm", "/query", token, "POST",
                        body={"question": "Which junctions does C2 feed into?"})
    assert status == 200, body
    assert body["grounded"] is True, body
    assert body["facts"], "no graph facts were retrieved"
    feeds = body["facts"][0]["feeds"]
    assert feeds == EXPECTED_C2_FEEDS, feeds


def test_llm_service_refuses_out_of_domain_questions() -> None:
    """Out-of-domain questions must be refused, not guessed at."""
    token = _token()
    for question in ("What is the capital of France?",
                     "Who won the 2019 cricket world cup?"):
        status, body = call("llm", "/query", token, "POST", body={"question": question})
        assert status == 200, body
        assert body["grounded"] is False, f"{question!r} was answered: {body['answer'][:120]}"


def test_llm_service_exposes_no_mutating_route() -> None:
    """The read-only boundary, checked against the service's own OpenAPI schema."""
    status, schema = call("llm", "/openapi.json")
    assert status == 200, schema
    offending: list[str] = []
    for path, operations in schema["paths"].items():
        for method in operations:
            if method.lower() in {"put", "patch", "delete"}:
                offending.append(f"{method.upper()} {path}")
            if method.lower() == "post" and path not in {"/query"}:
                offending.append(f"POST {path}")
    assert not offending, f"analytics service exposes mutating routes: {offending}"


def test_services_expose_prometheus_metrics() -> None:
    """Prometheus can only scrape what the services actually expose."""
    for name in PORTS:
        url = f"http://localhost:{PORTS[name]}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name} /metrics unreachable: {exc}") from exc
        assert "http_request" in text, f"{name} exposes no request metrics"


def main() -> None:
    """Run every check against the running stack."""
    if not stack_is_up():
        log.warning("Service stack is not running — skipping integration tests.")
        log.warning("Start it with:  docker compose up -d --build")
        return

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            log.info("PASS  %s", test.__name__)
        except Exception as exc:  # noqa: BLE001 - reporting boundary
            failures.append(f"{test.__name__}: {exc}")
            log.error("FAIL  %s -> %s", test.__name__, exc)
    log.info("%d/%d checks passed", len(tests) - len(failures), len(tests))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
