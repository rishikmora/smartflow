"""Print stylesheet and small HTML builders for the SmartFlow project dossier.

Kept beside :mod:`build_project_pdf` rather than inside it so the CSS - which is
long, and is really a design document - does not bury the logic that decides
what goes into the PDF.

Targets headless Chrome's print renderer, which does support CSS ``@page``
margin boxes; that is what produces the running header and the page numbers.
"""

from __future__ import annotations

import html

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
  @top-left { content: "SmartFlow - Multi-Agent Digital Twin for Adaptive Traffic Signal Control";
              font-family: Georgia, serif; font-size: 7.5pt; color: #8a8f98; }
  @bottom-right { content: counter(page) " / " counter(pages);
              font-family: Georgia, serif; font-size: 8pt; color: #8a8f98; }
  @bottom-left { content: "CMRCET - B.Tech CSE final-year project";
              font-family: Georgia, serif; font-size: 7.5pt; color: #a9aeb6; }
}
@page :first {
  margin: 0;
  @top-left { content: ""; }
  @bottom-right { content: ""; }
  @bottom-left { content: ""; }
}

:root {
  --ink: #161c26;
  --ink-2: #3d4653;
  --muted: #6d7581;
  --rule: #d9d5cd;
  --rule-2: #ebe8e2;
  --accent: #0f6d8c;
  --good: #17724a;
  --bad: #a3302a;
  --warn: #9a6508;
  --tint: #f4f2ed;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10pt;
  line-height: 1.58;
  color: var(--ink);
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1, h2, h3, h4, h5, h6, .ui {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

/* ---------------- cover ---------------- */
.cover {
  height: 297mm;
  padding: 30mm 24mm 22mm;
  display: flex;
  flex-direction: column;
  background: #12212c;
  color: #eef3f8;
  page-break-after: always;
}
.cover .kicker {
  font-family: Consolas, monospace; font-size: 8.5pt; letter-spacing: .22em;
  text-transform: uppercase; color: #7fb4cb; margin-bottom: 14mm;
}
.cover h1 {
  font-size: 34pt; line-height: 1.04; letter-spacing: -.02em;
  margin: 0 0 6mm; font-weight: 700;
}
.cover .sub {
  font-size: 13pt; line-height: 1.5; color: #b9cbd8; margin: 0 0 auto; max-width: 130mm;
}
.cover .facts { display: flex; flex-wrap: wrap; gap: 6mm 10mm; margin: 12mm 0 10mm; }
.cover .fact { min-width: 34mm; }
.cover .fact .k {
  font-family: Consolas, monospace; font-size: 7.5pt; letter-spacing: .14em;
  text-transform: uppercase; color: #7fb4cb;
}
.cover .fact .v {
  font-family: "Segoe UI", sans-serif; font-size: 17pt; font-weight: 600;
}
.cover .meta {
  border-top: 1px solid #2c4353; padding-top: 6mm; font-size: 9.5pt; color: #9fb5c4;
}
.cover .meta b { color: #eef3f8; font-weight: 600; }

/* ---------------- part dividers ---------------- */
.part {
  page-break-before: always;
  border-top: 2.5pt solid var(--ink);
  padding-top: 5mm;
  margin: 0 0 9mm;
}
.part .n {
  font-family: Consolas, monospace; font-size: 8.5pt; letter-spacing: .2em;
  text-transform: uppercase; color: var(--accent); display: block; margin-bottom: 2mm;
}
.part h1 { font-size: 21pt; margin: 0 0 2mm; letter-spacing: -.015em; line-height: 1.12; }
.part p { margin: 0; color: var(--muted); font-size: 10pt; max-width: 150mm; }

.section { page-break-before: always; }

/* ---------------- content headings ---------------- */
h2 { font-size: 14pt; margin: 8mm 0 2.5mm; letter-spacing: -.01em; line-height: 1.2;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 10pt; margin: 5mm 0 1.5mm; color: var(--ink-2); page-break-after: avoid; }
h5, h6 { font-size: 9.5pt; margin: 4mm 0 1.5mm; color: var(--muted);
         text-transform: uppercase; letter-spacing: .08em; page-break-after: avoid; }

p { margin: 0 0 3mm; orphans: 2; widows: 2; }
.lede { font-size: 10.5pt; color: var(--ink-2); }

ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }

a { color: var(--accent); text-decoration: none; }
strong, b { font-weight: 700; }

blockquote {
  margin: 3mm 0; padding: 2.5mm 5mm; border-left: 2.5pt solid var(--accent);
  background: var(--tint); color: var(--ink-2); font-style: italic;
}
blockquote p:last-child { margin-bottom: 0; }

code {
  font-family: Consolas, "SF Mono", monospace; font-size: 8.8pt;
  background: var(--tint); padding: .4mm 1.2mm; border-radius: 1mm; color: #2b3d47;
}
pre {
  font-family: Consolas, "SF Mono", monospace; font-size: 8.2pt; line-height: 1.45;
  background: #f7f6f2; border: .3pt solid var(--rule); border-radius: 1.5mm;
  padding: 3mm 4mm; white-space: pre-wrap; word-wrap: break-word;
  page-break-inside: avoid; margin: 0 0 3mm;
}
pre code { background: none; padding: 0; font-size: inherit; }

/* ---------------- tables ---------------- */
table { border-collapse: collapse; width: 100%; margin: 0 0 4mm;
        font-size: 8.6pt; page-break-inside: avoid; }
table.data { font-family: "Segoe UI", sans-serif; }
caption {
  caption-side: top; text-align: left; font-family: "Segoe UI", sans-serif;
  font-size: 8.5pt; font-weight: 600; color: var(--ink-2); padding-bottom: 1.5mm;
}
th, td { text-align: left; padding: 1.4mm 2mm; border-bottom: .3pt solid var(--rule-2);
         vertical-align: top; }
thead th {
  font-family: "Segoe UI", sans-serif; font-size: 7.8pt; text-transform: uppercase;
  letter-spacing: .07em; color: var(--muted); border-bottom: .8pt solid var(--rule);
  font-weight: 600;
}
tbody tr:nth-child(even) { background: #faf9f6; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums;
                 font-family: Consolas, monospace; }
tr.row-key td { background: #eef4f7; font-weight: 600; }

/* Wide CSVs (11 columns) overflow the A4 text block at body size and the last
   column gets clipped. Fixed layout plus a smaller face keeps every column on
   the page and lets long header names wrap instead of running off it. */
table.data.wide { font-size: 7pt; }
table.data.wide th, table.data.wide td { padding: .9mm 1.4mm; white-space: nowrap; }
table.data.wide thead th { font-size: 6.6pt; letter-spacing: .02em; }

/* A table taller than one page must be allowed to break, otherwise the whole
   block is pushed to the next page and still overflows. thead repeats. */
table.long { page-break-inside: auto; }
table.long tr { page-break-inside: avoid; }
table.long thead { display: table-header-group; }
.src { font-family: Consolas, monospace; font-size: 7.2pt; color: var(--muted);
       font-weight: 400; }

/* ---------------- figures ---------------- */
figure.chart { margin: 4mm 0 6mm; page-break-inside: avoid; text-align: center; }
figure.chart img { max-width: 100%; height: auto; border: .3pt solid var(--rule);
                   border-radius: 1.5mm; }
figcaption { font-family: "Segoe UI", sans-serif; font-size: 8.2pt; color: var(--muted);
             margin-top: 1.8mm; text-align: left; }

/* ---------------- callouts ---------------- */
.callout { border: .5pt solid var(--rule); border-left: 3pt solid var(--accent);
           background: #f8f7f4; padding: 3mm 4mm; margin: 4mm 0;
           page-break-inside: avoid; }
.callout.good { border-left-color: var(--good); }
.callout.bad { border-left-color: var(--bad); }
.callout.warn { border-left-color: var(--warn); }
.callout h4 { margin: 0 0 1.5mm; font-family: "Segoe UI", sans-serif; font-size: 9.5pt; }
.callout p:last-child { margin-bottom: 0; }

.badge { font-family: Consolas, monospace; font-size: 7.8pt; font-weight: 700;
         padding: .5mm 1.8mm; border-radius: 1mm; letter-spacing: .04em; }
.badge.met { background: #dcefe4; color: var(--good); }
.badge.fail { background: #f6dedc; color: var(--bad); }
.badge.none { background: #eeeae2; color: var(--muted); }

/* ---------------- contents ---------------- */
.toc { page-break-after: always; }
.toc h1 { font-size: 19pt; margin: 0 0 6mm; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { display: flex; align-items: baseline; gap: 3mm; padding: 1.6mm 0;
          border-bottom: .3pt dotted var(--rule);
          font-family: "Segoe UI", sans-serif; font-size: 9.5pt; }
.toc .n { font-family: Consolas, monospace; font-size: 8pt; color: var(--accent);
          min-width: 17mm; }
.toc .t { flex: 1; }
.toc .d { color: var(--muted); font-size: 8.5pt; max-width: 80mm; text-align: right; }

.note { font-size: 8.6pt; color: var(--muted); margin: 1.5mm 0 3mm; }
hr { border: 0; border-top: .3pt solid var(--rule); margin: 5mm 0; }
"""


def part(number: str, title: str, standfirst: str) -> str:
    """Render a part divider that starts a new page.

    Args:
        number: e.g. ``"Part III"``.
        title: the part title.
        standfirst: one-sentence description; may contain inline HTML.

    Returns:
        HTML for the divider.
    """
    return (f'<div class="part"><span class="n">{html.escape(number)}</span>'
            f"<h1>{html.escape(title)}</h1><p>{standfirst}</p></div>")


def callout(title: str, body: str, kind: str = "") -> str:
    """Render a boxed aside.

    Args:
        title: heading text.
        body: HTML body.
        kind: ``""``, ``"good"``, ``"bad"`` or ``"warn"``.

    Returns:
        HTML for the callout.
    """
    cls = f"callout {kind}".strip()
    return f'<div class="{cls}"><h4>{html.escape(title)}</h4>{body}</div>'


def toc(entries: list[tuple[str, str, str]]) -> str:
    """Render the table of contents.

    Args:
        entries: ``(number, title, description)`` triples.

    Returns:
        HTML for the contents page.
    """
    rows = "".join(
        f'<li><span class="n">{html.escape(n)}</span>'
        f'<span class="t">{html.escape(t)}</span>'
        f'<span class="d">{html.escape(d)}</span></li>'
        for n, t, d in entries
    )
    return f'<div class="toc"><h1>Contents</h1><ol>{rows}</ol></div>'
