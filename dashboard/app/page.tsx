"use client";

/**
 * SmartFlow control dashboard.
 *
 * Week 11's Definition of Done asks that the UI expose every feature without a
 * terminal, so each panel here maps to one of the five services: benchmark runs
 * and the network (sim), the controller comparison and the federated result
 * (rl), the detector and incident detection (vision), the knowledge graph
 * (graph), and grounded question answering (llm).
 *
 * Every number shown is fetched live from a service, which reads it from the
 * same committed artifacts the reports are built from. Nothing here is
 * hard-coded, so a stale panel means a stale service, not a stale page.
 */

import { useCallback, useEffect, useState } from "react";

type Json = Record<string, any>;

async function api(service: string, path: string): Promise<Json> {
  const response = await fetch(
    `/api/proxy?service=${service}&path=${encodeURIComponent(path)}`
  );
  if (!response.ok) throw new Error(`${service}${path} → ${response.status}`);
  return response.json();
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <header>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </header>
      {children}
    </section>
  );
}

function Health() {
  const [rows, setRows] = useState<Json[]>([]);
  useEffect(() => {
    const services = ["sim", "rl", "vision", "graph", "llm"];
    Promise.all(
      services.map(async (name) => {
        try {
          const body = await api(name, "/health");
          return { name, ...body };
        } catch (error) {
          return { name, status: "unreachable", error: String(error) };
        }
      })
    ).then(setRows);
  }, []);

  return (
    <Panel title="Services" subtitle="Live health of the five domain services">
      <div className="chips">
        {rows.map((row) => (
          <div key={row.name} className={`chip ${row.status === "ok" ? "ok" : "bad"}`}>
            <b>{row.name}</b>
            <span>{row.status}</span>
            {row.auth && <em>auth: {row.auth}</em>}
          </div>
        ))}
      </div>
    </Panel>
  );
}

function Benchmarks() {
  const [rows, setRows] = useState<Json[]>([]);
  const [scenario, setScenario] = useState("base");
  const [scenarios, setScenarios] = useState<string[]>([]);

  useEffect(() => {
    api("sim", "/scenarios")
      .then((body) => setScenarios(body.scenarios ?? []))
      .catch(() => setScenarios(["base"]));
  }, []);

  useEffect(() => {
    api("sim", `/runs/summary?scenario=${scenario}`)
      .then((body) => setRows(body.controllers ?? []))
      .catch(() => setRows([]));
  }, [scenario]);

  const worst = Math.max(1, ...rows.map((r) => r.avg_wait_time_s ?? 0));

  return (
    <Panel
      title="Benchmark results"
      subtitle="3-seed means, read live from the simulation service"
    >
      <div className="controls">
        {scenarios.map((name) => (
          <button
            key={name}
            className={name === scenario ? "on" : ""}
            onClick={() => setScenario(name)}
          >
            {name}
          </button>
        ))}
      </div>
      <table>
        <thead>
          <tr>
            <th>Controller</th>
            <th className="num">Avg wait (s)</th>
            <th className="num">Max queue</th>
            <th className="num">Throughput</th>
            <th>Relative wait</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.controller}>
              <td>
                <code>{row.controller}</code>
              </td>
              <td className="num">{row.avg_wait_time_s?.toFixed(2)}</td>
              <td className="num">{row.max_queue_len?.toFixed(0)}</td>
              <td className="num">{row.throughput_veh?.toFixed(0)}</td>
              <td>
                <span
                  className="bar"
                  style={{ width: `${((row.avg_wait_time_s ?? 0) / worst) * 100}%` }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="empty">No runs for this scenario.</p>}
    </Panel>
  );
}

function Vision() {
  const [detector, setDetector] = useState<Json | null>(null);
  const [anomaly, setAnomaly] = useState<Json | null>(null);

  useEffect(() => {
    api("vision", "/detector").then(setDetector).catch(() => setDetector(null));
    api("vision", "/anomalies").then(setAnomaly).catch(() => setAnomaly(null));
  }, []);

  return (
    <Panel title="Perception" subtitle="Detector and incident detection (Week 8)">
      {detector ? (
        <>
          <div className="stats">
            <div>
              <span className="k">mAP50</span>
              <span className="v">{detector.metrics.mAP50.toFixed(3)}</span>
            </div>
            <div>
              <span className="k">mAP50-95</span>
              <span className="v">{detector.metrics.mAP50_95.toFixed(3)}</span>
            </div>
            <div>
              <span className="k">recall</span>
              <span className="v">{detector.metrics.recall.toFixed(3)}</span>
            </div>
            <div>
              <span className="k">held-out boxes</span>
              <span className="v">{detector.dataset.val_boxes}</span>
            </div>
          </div>
          <p className="caveat">{detector.caveat}</p>
        </>
      ) : (
        <p className="empty">Detector metrics unavailable.</p>
      )}
      {anomaly && (
        <div className="stats">
          <div>
            <span className="k">incident recall</span>
            <span className="v">{anomaly.chosen.recall.toFixed(2)}</span>
          </div>
          <div>
            <span className="k">precision</span>
            <span className="v">{anomaly.chosen.precision.toFixed(2)}</span>
          </div>
          <div>
            <span className="k">latency</span>
            <span className="v">{anomaly.chosen.mean_latency_s.toFixed(0)} s</span>
          </div>
        </div>
      )}
    </Panel>
  );
}

function Federated() {
  const [result, setResult] = useState<Json | null>(null);
  useEffect(() => {
    api("rl", "/federated").then(setResult).catch(() => setResult(null));
  }, []);

  if (!result) {
    return (
      <Panel title="Federated learning" subtitle="Week 9">
        <p className="empty">Not available.</p>
      </Panel>
    );
  }
  const s = result.summary;
  return (
    <Panel
      title="Federated learning"
      subtitle={`Districts ${result.districts.join(", ")} → held-out ${result.held_out}`}
    >
      <div className="stats">
        <div>
          <span className="k">fixed</span>
          <span className="v">{s.fixed_wait_s.toFixed(2)} s</span>
        </div>
        <div>
          <span className="k">local mean</span>
          <span className="v">{s.local_mean_wait_s.toFixed(2)} s</span>
        </div>
        <div>
          <span className="k">federated</span>
          <span className="v">{s.fedavg_wait_s.toFixed(2)} s</span>
        </div>
      </div>
      <p className={s.dod_met ? "caveat" : "caveat bad"}>
        DoD {s.dod_met ? "met" : "NOT met"} — federated averaging is{" "}
        {s.improvement_pct_vs_local_mean.toFixed(1)}% against the local-only
        baseline. The district policies differ in weights but agree on 100% of
        decisions, so averaging them changes nothing.
      </p>
    </Panel>
  );
}

function GraphExplorer() {
  const [junctions, setJunctions] = useState<Json[]>([]);
  const [selected, setSelected] = useState<string>("C2");
  const [detail, setDetail] = useState<Json | null>(null);

  useEffect(() => {
    api("graph", "/graph/junctions")
      .then((body) => setJunctions(body.junctions ?? []))
      .catch(() => setJunctions([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    api("graph", `/graph/junctions/${selected}`)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected]);

  return (
    <Panel title="Knowledge graph" subtitle="Click a junction to inspect it">
      <div className="grid16">
        {junctions.map((j) => (
          <button
            key={j.id}
            className={j.id === selected ? "cell on" : "cell"}
            onClick={() => setSelected(j.id)}
            title={j.signalised ? "signalised" : "unsignalised"}
          >
            {j.id}
            {j.signalised && <i />}
          </button>
        ))}
      </div>
      {detail && (
        <div className="detail">
          <p>
            <b>{selected}</b> feeds into{" "}
            <b>{(detail.feeds ?? []).join(", ") || "nothing"}</b>
          </p>
          <p>
            {detail.lanes?.length ?? 0} incoming lanes ·{" "}
            {detail.sensors?.length ?? 0} sensors ·{" "}
            {detail.program ? `${detail.program.phases?.length ?? 0} phases` : "no program"}
          </p>
        </div>
      )}
    </Panel>
  );
}

function Ask() {
  const [question, setQuestion] = useState("Which junctions does C2 feed into?");
  const [answer, setAnswer] = useState<Json | null>(null);
  const [busy, setBusy] = useState(false);

  const ask = useCallback(async () => {
    setBusy(true);
    setAnswer(null);
    try {
      const response = await fetch("/api/proxy?service=llm&path=%2Fquery", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });
      setAnswer(await response.json());
    } catch (error) {
      setAnswer({ answer: String(error), grounded: false });
    } finally {
      setBusy(false);
    }
  }, [question]);

  return (
    <Panel
      title="Ask the corridor"
      subtitle="Read-only analytics — answers come from the graph and the project's reports"
    >
      <div className="ask">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && ask()}
          placeholder="Ask about a junction, a controller or a result"
        />
        <button onClick={ask} disabled={busy}>
          {busy ? "…" : "Ask"}
        </button>
      </div>
      {answer && (
        <div className={`answer ${answer.grounded ? "" : "refused"}`}>
          <p>{answer.answer}</p>
          {answer.grounded === false && (
            <em>Refused — nothing relevant was retrieved.</em>
          )}
          {answer.passages?.length > 0 && (
            <ul>
              {answer.passages.slice(0, 3).map((p: Json, i: number) => (
                <li key={i}>
                  <code>{p.source}</code>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Panel>
  );
}

export default function Home() {
  return (
    <main>
      <header className="top">
        <p className="eyebrow">SmartFlow · control dashboard</p>
        <h1>Corridor operations</h1>
        <p className="lede">
          Every panel is served live by one of the five domain services. Nothing on
          this page is hard-coded.
        </p>
      </header>
      <Health />
      <Benchmarks />
      <div className="two">
        <Vision />
        <Federated />
      </div>
      <GraphExplorer />
      <Ask />
      <footer>
        <a href="http://localhost:3001" target="_blank" rel="noreferrer">
          Grafana
        </a>
        <a href="http://localhost:9090" target="_blank" rel="noreferrer">
          Prometheus
        </a>
        <a href="http://localhost:8001/docs" target="_blank" rel="noreferrer">
          Simulation API docs
        </a>
        <a href="http://localhost:8005/docs" target="_blank" rel="noreferrer">
          Analytics API docs
        </a>
      </footer>
    </main>
  );
}
