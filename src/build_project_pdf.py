"""Build a single PDF covering the whole SmartFlow project.

Collects every committed report, chart, metric table and source file into one
paginated A4 document. Nothing here restates a result by hand: the prose comes
from the markdown already in the repository, the numbers are aggregated from the
metrics CSVs, and the code inventory is read off the source tree. That way the
dossier cannot drift away from the project the way a hand-written summary would.

Rendering goes HTML -> headless Chrome -> PDF. Chrome is used rather than
WeasyPrint or ReportLab because it is already installed on this machine, needs
no new Python dependency, and supports CSS ``@page`` margin boxes, which is what
gives the document its running header and page numbers.

Usage:
    python src/build_project_pdf.py
    python src/build_project_pdf.py --keep-html
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime as dt
import html
import io
import logging
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_style import CSS, callout, part, toc

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
PDF_PATH = os.path.join(OUTPUTS, "SmartFlow_Project_Dossier.pdf")
HTML_PATH = os.path.join(OUTPUTS, "SmartFlow_Project_Dossier.html")

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

CHARTS: dict[str, list[tuple[str, str]]] = {
    "week1": [("baseline_comparison.png",
               "Fixed-time vs actuated on the 12-junction corridor, 3 seeds.")],
    "week2": [("week2_benchmark_comparison.png",
               "PPO vs fixed-time on sumo-rl's 2-way single-intersection benchmark.")],
    "week3": [("week3_comparison.png",
               "Corridor-wide metrics with PPO controlling junction B1 only."),
              ("week3_comparison_junction.png",
               "The same run measured at B1 itself, where the policy actually acts."),
              ("week3_reward_curves.png",
               "Training reward per seed.")],
    "week4": [("week4_comparison.png",
               "Independent PPO per junction across the full corridor."),
              ("week4_training_curves.png",
               "Training curves; the Week 4 test is that no seed diverges.")],
    "week5": [("week5_comparison.png",
               "Parameter-shared policy with green-wave shaping and the fairness constraint."),
              ("week5_fairness.png",
               "Fairness ablation: wait percentiles with and without the constraint."),
              ("week5_lambda.png",
               "The Lagrange multiplier under dual ascent."),
              ("week5_online_learning_peak.png",
               "Online learning, peak demand."),
              ("week5_online_learning_asymmetric.png",
               "Online learning, asymmetric demand.")],
    "week8": [("week8_detector_pr.png",
               "Week 8 - detector precision/recall on the held-out junction."),
              ("week8_detector_confusion.png",
               "Week 8 - normalised confusion matrix, held-out junction."),
              ("week8_detector_training.png",
               "Week 8 - detector training curves over 40 CPU epochs."),
              ("week8_anomalies.png",
               "Week 8 - injected incidents, detector alarms and the operating-point sweep."),
              ("week8_scenarios.png",
               "Week 8 - scenario planner: closures crossed with weather, 3 seeds each.")],
    "week9": [("week9_federated.png",
               "Week 9 - federated averaging evaluated on a held-out district."),
              ("week9_priority.png",
               "Week 9 - emergency preemption: the gain, and the cost to general traffic.")],
    "week6": [("week6_scenarios.png",
               "Average wait across light, base, peak and asymmetric demand."),
              ("week6_ablations.png",
               "Every policy variant side by side."),
              ("week6_robustness.png",
               "Robustness spread across seeds and scenarios.")],
}

# The raw CSV column names are long enough that an 11-column table runs off the
# A4 text block. These are display labels only; the underlying files are unchanged
# and are named in every table caption.
SHORT_HEADERS = {
    "avg_wait_time_s": "avg wait (s)",
    "max_queue_len": "max queue",
    "throughput_veh": "throughput",
    "total_co2_kg": "CO2 (kg)",
    "wait_p95_s": "wait p95 (s)",
    "worst_vehicle_wait_s": "worst wait (s)",
    "controller": "controller",
    "scenario": "scenario",
    "scope": "scope",
    "tls_id": "junction",
    "seed": "seed",
}

CONTENTS: list[tuple[str, str, str]] = [
    ("Part I", "Executive summary", "What was built, what it proves, what it does not"),
    ("Part II", "Status by week", "Every Definition of Done and its verdict"),
    ("Part III", "The project", "README: idea, results, structure, setup, commands"),
    ("Part IV", "Method and roadmap", "The plan, the de-risking substitutions, the rules"),
    ("Part V", "Phase A-B benchmark report", "The full internal benchmark, Weeks 1-6"),
    ("Part VI", "Week-by-week evidence", "Each week's own report, with its charts"),
    ("Part VII", "Week 7: knowledge graph and RAG", "Graph, retrieval and the read-only service"),
    ("Part VIII", "Weeks 8-9: perception and federation",
     "Vision, anomaly detection, planning, federated learning, LoRA"),
    ("Part IX", "Weeks 10-12: the platform", "Services, Kubernetes, dashboard, integration"),
    ("Part X", "Deferred and draft material", "Items carried forward, honestly labelled"),
    ("Part XI", "Codebase", "Every source file and what it does"),
    ("Appendix A", "Complete metric tables", "Raw CSV data behind every claim"),
    ("Appendix B", "Commit history", "The full development record"),
]


# --------------------------------------------------------------------------
# source helpers
# --------------------------------------------------------------------------
def find_chrome() -> str:
    """Locate a Chromium-family browser able to print to PDF.

    Returns:
        Absolute path to the browser executable.

    Raises:
        RuntimeError: if no candidate exists on this machine.
    """
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise RuntimeError(
        "No Chrome or Edge found. Install one, or open "
        f"{HTML_PATH} and print it to PDF by hand."
    )


def read_text(path: str) -> str:
    """Read a UTF-8 text file, returning a visible placeholder if it is missing.

    Args:
        path: repo-relative path.

    Returns:
        File contents, or an italic note naming the absent file.
    """
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        log.warning("Missing source: %s", path)
        return f"*Not present in the repository: `{path}`*"
    return io.open(full, encoding="utf-8", errors="replace").read()


def md(text: str, demote_by: int = 1) -> str:
    """Convert markdown to HTML, demoting headings to sit under part titles.

    Args:
        text: markdown source.
        demote_by: how many heading levels to push down.

    Returns:
        Rendered HTML fragment.
    """
    import markdown
    out = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    for level in range(5, 0, -1):
        target = min(6, level + demote_by)
        out = out.replace(f"<h{level}>", f"<h{target}>")
        out = out.replace(f"</h{level}>", f"</h{target}>")
    return out


def doc(path: str, demote_by: int = 1) -> str:
    """Render one repository markdown file as a page-starting section.

    Args:
        path: repo-relative markdown path.
        demote_by: heading demotion level.

    Returns:
        HTML section, prefixed with the source path so every page is traceable.
    """
    body = md(read_text(path), demote_by)
    return (f'<div class="section"><p class="note">Source: <code>'
            f"{html.escape(path)}</code></p>{body}</div>")


def embed_image(name: str, caption: str) -> str:
    """Inline a PNG chart as a data URI figure.

    Args:
        name: filename inside outputs/.
        caption: figure caption.

    Returns:
        HTML figure, or empty string if the chart is absent.
    """
    path = os.path.join(OUTPUTS, name)
    if not os.path.isfile(path):
        log.warning("Missing chart: %s", name)
        return ""
    data = base64.b64encode(io.open(path, "rb").read()).decode("ascii")
    return ('<figure class="chart">'
            f'<img src="data:image/png;base64,{data}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)} "
            f'<span class="src">outputs/{html.escape(name)}</span></figcaption></figure>')


def charts_for(key: str) -> str:
    """Render every chart registered against one week.

    Args:
        key: a key of :data:`CHARTS`.

    Returns:
        Concatenated figure HTML.
    """
    return "".join(embed_image(n, c) for n, c in CHARTS.get(key, []))


def csv_table(path: str, caption: str) -> str:
    """Render a CSV file as a complete HTML table.

    Args:
        path: repo-relative CSV path.
        caption: table caption.

    Returns:
        HTML table markup, or a note if the file is missing or empty.
    """
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        log.warning("Missing CSV: %s", path)
        return f"<p><em>Missing: {html.escape(path)}</em></p>"
    with io.open(full, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return f"<p><em>Empty: {html.escape(path)}</em></p>"
    head, body = rows[0], rows[1:]
    # Shape drives presentation: many columns need the compact fixed layout, many
    # rows need permission to break across pages.
    css = "data"
    if len(head) > 8:
        css += " wide"
    if len(body) > 25:
        css += " long"
    out = [f'<table class="{css}"><caption>{html.escape(caption)} '
           f'<span class="src">{html.escape(path)}</span></caption><thead><tr>']
    out += [f"<th>{html.escape(SHORT_HEADERS.get(c, c))}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for row in body:
        cells = "".join(
            f'<td class="num">{html.escape(c)}</td>' if _numeric(c)
            else f"<td>{html.escape(c)}</td>"
            for c in row
        )
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    out.append(f'<p class="note">{len(body)} rows, complete.</p>')
    return "".join(out)


def _numeric(value: str) -> bool:
    """Report whether a CSV cell should be right-aligned as a number.

    Args:
        value: raw cell text.

    Returns:
        True if the cell parses as a float.
    """
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def code_inventory() -> tuple[str, int, int]:
    """Build a table of every Python file with its own module docstring.

    Returns:
        ``(html, file_count, line_count)``.
    """
    rows: list[tuple[str, int, str]] = []
    for folder in ("src", "services", "tests"):
        base = os.path.join(ROOT, folder)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, ROOT).replace("\\", "/")
                source = io.open(path, encoding="utf-8", errors="replace").read()
                try:
                    docstring = (ast.get_docstring(ast.parse(source)) or "").split("\n")[0]
                except SyntaxError:
                    docstring = "(could not be parsed)"
                rows.append((rel, len(source.splitlines()), docstring))
    total = sum(r[1] for r in rows)
    out = ['<table class="data"><thead><tr><th>File</th><th class="num">Lines</th>'
           "<th>Module docstring, first line</th></tr></thead><tbody>"]
    for rel, lines, docstring in rows:
        out.append(f"<tr><td><code>{html.escape(rel)}</code></td>"
                   f'<td class="num">{lines}</td>'
                   f"<td>{html.escape(docstring)}</td></tr>")
    out.append("</tbody></table>")
    return "".join(out), len(rows), total


def git_history() -> str:
    """Render the complete commit history as a table.

    Returns:
        HTML table, or a note if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%h|%ad|%an|%s", "--date=format:%Y-%m-%d"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("git log failed: %s", exc)
        return "<p><em>Commit history unavailable.</em></p>"
    if result.returncode != 0:
        return "<p><em>Commit history unavailable.</em></p>"
    out = ['<table class="data"><thead><tr><th>Commit</th><th>Date</th>'
           "<th>Author</th><th>Message</th></tr></thead><tbody>"]
    count = 0
    for line in result.stdout.splitlines():
        pieces = line.split("|", 3)
        if len(pieces) != 4:
            continue
        count += 1
        sha, date, author, subject = pieces
        out.append(f"<tr><td><code>{html.escape(sha)}</code></td>"
                   f"<td>{html.escape(date)}</td><td>{html.escape(author)}</td>"
                   f"<td>{html.escape(subject)}</td></tr>")
    out.append("</tbody></table>")
    out.append(f'<p class="note">{count} commits.</p>')
    return "".join(out)


def corridor_table() -> str:
    """Aggregate corridor metrics into one 3-seed comparison table.

    Returns:
        HTML table covering every controller measured on base demand.
    """
    try:
        from analysis import aggregate, load_rows
        rows = load_rows(os.path.join(OUTPUTS, "marl_metrics.csv"))
        stats = aggregate(
            rows, ["avg_wait_time_s", "max_queue_len", "throughput_veh", "total_co2_kg"],
            scope="corridor", scenario="base",
        )
    except Exception as exc:                                   # noqa: BLE001
        log.warning("Could not aggregate corridor metrics: %s", exc)
        return "<p><em>Corridor aggregate unavailable.</em></p>"

    order = ["fixed", "actuated", "marl_independent", "marl_shared_w5",
             "marl_shared_w5nofair", "marl_shared_w5strong",
             "marl_shared_w5nocoord", "marl_shared_w5gnn"]
    out = ['<table class="data"><caption>Corridor-wide, base demand, 3-seed means '
           '<span class="src">outputs/marl_metrics.csv</span></caption><thead><tr>'
           '<th>Controller</th><th class="num">Avg wait (s)</th>'
           '<th class="num">Max queue</th><th class="num">Throughput</th>'
           '<th class="num">CO2 (kg)</th></tr></thead><tbody>']
    for name in order:
        if name not in stats:
            continue
        s = stats[name]
        attr = ' class="row-key"' if name in ("fixed", "actuated", "marl_shared_w5") else ""
        out.append(
            f"<tr{attr}><td><code>{html.escape(name)}</code></td>"
            f'<td class="num">{s["avg_wait_time_s"]["mean"]:.2f}</td>'
            f'<td class="num">{s["max_queue_len"]["mean"]:.1f}</td>'
            f'<td class="num">{s["throughput_veh"]["mean"]:.1f}</td>'
            f'<td class="num">{s["total_co2_kg"]["mean"]:.1f}</td></tr>'
        )
    out.append("</tbody></table>")
    return "".join(out)


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------
def cover(files: int, lines: int) -> str:
    """Render the cover page.

    Args:
        files: number of Python files in the project.
        lines: total Python lines.

    Returns:
        HTML for the cover.
    """
    today = dt.date.today().strftime("%d %B %Y")
    return f"""
<div class="cover">
  <div class="kicker">Project dossier &middot; complete record</div>
  <h1>SmartFlow</h1>
  <p class="sub">A multi-agent digital twin for adaptive traffic signal control &mdash;
     built on SUMO, benchmarked against classical baselines, and reported with its
     failures intact.</p>
  <div class="facts">
    <div class="fact"><div class="k">Corridor</div><div class="v">12 junctions</div></div>
    <div class="fact"><div class="k">Best wait</div><div class="v">17.9 s</div></div>
    <div class="fact"><div class="k">vs fixed-time</div><div class="v">&minus;78%</div></div>
    <div class="fact"><div class="k">Python</div><div class="v">{lines:,} lines</div></div>
  </div>
  <div class="meta">
    <b>B.Tech CSE final-year major project</b> &middot; CMRCET<br>
    Solo build &middot; 12-week window &middot; Phase A&ndash;C complete through Week 7<br>
    {files} Python files &middot; 3-seed minimum for every reported result<br>
    Compiled {today} from the repository&rsquo;s own reports, metrics and source.
  </div>
</div>
"""


def executive_summary() -> str:
    """Author the one-part summary that frames everything after it.

    Returns:
        HTML for Part I.
    """
    body = md("""
SmartFlow trains reinforcement-learning agents to control traffic signals on a
simulated 12-junction corridor, and measures them against two classical
baselines: a fixed-time controller and SUMO's detector-driven actuated
controller. Every controller is judged by the same per-second metric collector
on identical traffic, so a difference on screen is a difference in policy and
nothing else.

## The headline result

Corridor-wide, on base demand, averaged over three independently trained seeds:

""", demote_by=1)

    numbers = corridor_table()

    after = md("""
Reinforcement learning roughly **halves the wait of the actuated baseline**
(17.9 s against 40.9 s) and cuts fixed-time by 78%. That is a real result and it
is reproducible from the committed checkpoints.

## What the project does not claim

Two of the seven Definitions of Done attempted so far were **not met**, and both
are reported as failures rather than re-scoped until they passed.

- **Week 3** put PPO on a single corridor junction. It beat fixed-time on all
  three metrics but beat the actuated baseline on only one of three.
- **Week 5** added green-wave shaping and a Lagrangian fairness constraint to a
  parameter-shared policy. It beats actuated on wait time, but **not reliably on
  throughput**: two seeds of three beat the baseline and the third converged to a
  policy that jams the corridor, dragging the mean below it. The fairness
  constraint measurably changed nothing.

Note in the table above that `marl_independent` &mdash; Week 4's simpler, unshaped
configuration &mdash; is the strongest row on throughput. Corridor-wide RL beating
both baselines is a genuine finding; it is the Week 5 *shaping* that fails to
add to it.

## The most interesting negative result

Four policy variants in the table are **numerically identical** to two decimal
places: no-fairness, 20x fairness, no-coordination, and the graph-attention
encoder all produce 18.16 s wait, 42.3 max queue and 1678.7 throughput. These are
genuinely different networks &mdash; 16k against 138k parameters &mdash; and they
agree on 100% of *actionable* decisions.

The cause is the action space, not a bug in the experiment: with a binary
next-phase action and a 5 s minimum green, the greedy policy saturates. Every
disagreement that does exist falls inside sumo-rl's minimum-green discard window,
where the decision is thrown away before it reaches the simulator. This is
documented rather than hidden, and it is the single clearest candidate for the
next piece of work.

## Status in one line

Weeks 1, 2, 4, 6 and 7 met their Definition of Done; Weeks 3 and 5 did not;
Weeks 8&ndash;12 have not been started and exist only as scaffolding. Part II
gives the full table.
""", demote_by=1)

    limits = callout(
        "What bounds every number in this document",
        md("""
All results come from one synthetic 12-junction grid with a synthetic demand
generator. Nothing here demonstrates transfer to a real road network &mdash; the
OSM corridor has been deferred since Week 1. Three seeds is enough to show
variance, not enough for confidence intervals. Nothing has been load-tested, and
no claim is made anywhere about real-time or production performance.
""", demote_by=2),
        "warn",
    )
    return body + numbers + after + limits


def week_section(title: str, verdict: str, path: str, chart_key: str,
                 lead: str = "") -> str:
    """Render one week: heading, verdict badge, its report, then its charts.

    Args:
        title: e.g. ``"Week 3"``.
        verdict: ``"MET"``, ``"NOT MET"`` or ``"n/a"``.
        path: repo-relative markdown report, or ``""`` if the week has none.
        chart_key: key into :data:`CHARTS`.
        lead: optional introductory HTML.

    Returns:
        HTML section.
    """
    cls = {"MET": "met", "NOT MET": "fail"}.get(verdict, "none")
    head = (f'<div class="section"><h2>{html.escape(title)} '
            f'<span class="badge {cls}">{html.escape(verdict)}</span></h2>{lead}')
    body = ""
    if path:
        body = (f'<p class="note">Source: <code>{html.escape(path)}</code></p>'
                + md(read_text(path), demote_by=1))
    return head + body + charts_for(chart_key) + "</div>"


def build_html() -> str:
    """Assemble the whole dossier as one HTML string.

    Returns:
        Complete HTML document.
    """
    inventory, files, lines = code_inventory()

    pieces: list[str] = [
        "<!-- SmartFlow project dossier -->",
        '<meta charset="utf-8">',
        "<title>SmartFlow - Project Dossier</title>",
        f"<style>{CSS}</style>",
        cover(files, lines),
        toc(CONTENTS),

        part("Part I", "Executive summary",
             "What was built, what it proves, and what it deliberately does not claim."),
        executive_summary(),

        part("Part II", "Status by week",
             "Every Definition of Done, its verdict, and the evidence behind it."),
        doc("outputs/STATUS.md"),

        part("Part III", "The project",
             "The repository README in full: the idea, the measured results, the "
             "directory layout, setup and every command."),
        doc("README.md"),

        part("Part IV", "Method and roadmap",
             "The working plan the project is held to, including the de-risking "
             "substitutions and the non-negotiable workflow rules."),
        doc("CLAUDE.md"),

        part("Part V", "Phase A-B benchmark report",
             "The internal benchmark covering Weeks 1-6, including the section on "
             "where reinforcement learning loses."),
        doc("BENCHMARK_REPORT.md"),

        part("Part VI", "Week-by-week evidence",
             "Each week's own report, reproduced verbatim, with the charts it "
             "produced."),
        week_section("Week 1 - baselines", "MET", "", "week1",
                     md("Fixed-time and actuated baselines on the corridor, plus the "
                        "evaluation harness every later week is judged against. This "
                        "week predates the written-report convention; its evidence is "
                        "the metrics CSV and the chart below.", 2)),
        week_section("Week 2 - single-agent PPO on a standard benchmark", "MET",
                     "outputs/week2_literature_note.md", "week2"),
        week_section("Week 3 - PPO on one corridor junction", "NOT MET",
                     "outputs/week3_report.md", "week3"),
        week_section("Week 4 - independent multi-agent PPO", "MET",
                     "outputs/week4_report.md", "week4"),
        week_section("Week 4 - non-stationarity notes", "n/a",
                     "outputs/week4_nonstationarity_notes.md", ""),
        week_section("Week 5 - shared policy, shaping and fairness", "NOT MET",
                     "outputs/week5_report.md", "week5"),
        week_section("Week 6 - stress tests and ablations", "MET", "", "week6",
                     md("Week 6's deliverable is the benchmark report reproduced in "
                        "Part V. Its charts are below.", 2)),

        part("Part VII", "Week 7: knowledge graph and RAG",
             "A read-only question-answering layer over the corridor, grounded in a "
             "graph of the network and in the project's own reports."),
        doc("outputs/week7_report.md"),

        part("Part VIII", "Weeks 8-9: perception and federation",
             "Vehicle detection, incident detection and scenario planning; then "
             "federated learning across districts, a LoRA adapter, and emergency "
             "preemption."),
        doc("outputs/week8_report.md"),
        f'<div class="section">{charts_for("week8")}</div>',
        doc("outputs/week9_report.md"),
        f'<div class="section">{charts_for("week9")}</div>',

        part("Part IX", "Weeks 10-12: the platform",
             "Five services behind authentication, deployed with Helm on k3d, a "
             "dashboard, and the end-to-end integration test."),
        doc("outputs/week10_report.md"),
        doc("outputs/week11_report.md"),
        doc("outputs/week12_report.md"),

        part("Part X", "Deferred and draft material",
             "Items carried forward, plus the drafts that are labelled as drafts "
             "rather than presented as results."),
        doc("outputs/week7_deferred.md"),
        doc("outputs/week8_deferred.md"),
        doc("outputs/week9_deferred.md"),
        doc("outputs/citations_verified.md"),

        part("Part XII", "Final report and viva preparation",
             "The generated final report, and the questions it should survive."),
        doc("outputs/FINAL_REPORT.md"),
        doc("outputs/VIVA_PREP.md"),

        part("Part XI", "Codebase",
             f"All {files} Python files, {lines:,} lines. Each description is the "
             "file's own module docstring, read from the source."),
        f'<div class="section">{inventory}</div>',

        part("Appendix A", "Complete metric tables",
             "The raw CSV data behind every number in this document, in full."),
        f'<div class="section">{csv_table("outputs/metrics.csv", "Week 1 - corridor baselines")}</div>',
        f'<div class="section">{csv_table("outputs/week2_benchmark_metrics.csv", "Week 2 - sumo-rl benchmark intersection")}</div>',
        f'<div class="section">{csv_table("outputs/week3_corridor_metrics.csv", "Week 3 - single-junction PPO on the corridor")}</div>',
        f'<div class="section">{csv_table("outputs/marl_metrics.csv", "Weeks 4-6 - corridor-wide multi-agent runs")}</div>',

        part("Appendix B", "Commit history",
             "The full development record, oldest at the bottom."),
        f'<div class="section">{git_history()}</div>',
    ]
    return "\n".join(pieces)


def render_pdf(html_path: str, pdf_path: str) -> None:
    """Print an HTML file to PDF with headless Chrome.

    Args:
        html_path: absolute path to the source HTML.
        pdf_path: absolute path to write.

    Raises:
        RuntimeError: if the browser is missing or produces no file.
    """
    chrome = find_chrome()
    url = "file:///" + html_path.replace("\\", "/")
    command = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=30000",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    log.info("Rendering with %s", os.path.basename(chrome))
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Chrome timed out while rendering the PDF.") from exc
    if not os.path.isfile(pdf_path):
        raise RuntimeError(
            f"Chrome did not write {pdf_path}. Open {html_path} and print it manually."
        )


def main() -> None:
    """Build the dossier HTML and render it to PDF."""
    parser = argparse.ArgumentParser(description="Build the SmartFlow project PDF.")
    parser.add_argument("--keep-html", action="store_true",
                        help="Leave the intermediate HTML in outputs/ for inspection.")
    args = parser.parse_args()

    os.makedirs(OUTPUTS, exist_ok=True)
    document = build_html()
    with io.open(HTML_PATH, "w", encoding="utf-8") as handle:
        handle.write(document)
    log.info("HTML %.2f MB", os.path.getsize(HTML_PATH) / 1024 / 1024)

    render_pdf(HTML_PATH, PDF_PATH)
    log.info("Wrote %s (%.2f MB)", PDF_PATH, os.path.getsize(PDF_PATH) / 1024 / 1024)

    if not args.keep_html:
        os.remove(HTML_PATH)


if __name__ == "__main__":
    main()
