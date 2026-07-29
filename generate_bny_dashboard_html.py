"""
generate_bny_dashboard_html.py
--------------------------------
Builds a self-contained HTML version of the
"BNY Managed Data Services - Executive Service Health" dashboard, for people
without PowerPoint. It reuses the exact same data logic as
generate_bny_dashboard.py (same metrics, scoring, watchlist, themes) and renders
the two charts as inline SVG so the file works fully offline - no internet,
no PowerPoint, no extra software. Just double-click the .html to open it in any
browser, and use the browser's Print -> Save as PDF for a 2-page report.

Run:
    python generate_bny_dashboard_html.py

Output:
    BNY_Executive_Dashboard_Services_<DDMMYYYY>.html written to the source dir.
"""

import os

import pandas as pd

from generate_bny_dashboard import (
    CONFIG,
    RAG_LABEL,
    compute_metrics,
    find_latest_source_file,
    load_cases,
    parse_asof_from_filename,
)

RAG_HEX = {"G": "#16a34a", "A": "#f59e0b", "R": "#dc2626"}
BAND_HEX = {"GREEN": "#16a34a", "AMBER": "#f59e0b", "RED": "#dc2626"}


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------
# INLINE SVG CHARTS (no external libraries -> works offline)
# --------------------------------------------------------------------------
def svg_line_chart(week_labels, series_p1, series_p2, cap):
    w, h = 560, 300
    pad_l, pad_r, pad_t, pad_b = 45, 20, 20, 40
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    max_v = max([cap] + series_p1 + series_p2) or 1
    max_v = max(max_v, 1)

    def x(i):
        return pad_l + (plot_w * i / max(1, len(week_labels) - 1))

    def y(v):
        return pad_t + plot_h - (plot_h * v / max_v)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="MTTR trend chart">']
    # gridlines + y axis labels
    for g in range(5):
        gv = max_v * g / 4
        gy = y(gv)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" font-size="10" fill="#5b6370" text-anchor="end">{gv:.0f}</text>')
    # x labels
    for i, lbl in enumerate(week_labels):
        parts.append(f'<text x="{x(i):.1f}" y="{h - pad_b + 18}" font-size="10" fill="#5b6370" text-anchor="middle">{esc(lbl)}</text>')

    def polyline(series, color):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series))
        out = [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>']
        for i, v in enumerate(series):
            out.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="{color}"/>')
        return "".join(out)

    parts.append(polyline(series_p2, "#f59e0b"))
    parts.append(polyline(series_p1, "#dc2626"))
    # legend
    parts.append(f'<rect x="{pad_l}" y="4" width="10" height="10" fill="#dc2626"/><text x="{pad_l + 15}" y="13" font-size="10" fill="#1f2937">P1 (days)</text>')
    parts.append(f'<rect x="{pad_l + 90}" y="4" width="10" height="10" fill="#f59e0b"/><text x="{pad_l + 105}" y="13" font-size="10" fill="#1f2937">P2 (days)</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bar_chart(categories, values):
    w, h = 560, 300
    pad_l, pad_r, pad_t, pad_b = 45, 20, 20, 40
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    max_v = max(values + [1])
    n = len(categories)
    slot = plot_w / n
    bar_w = slot * 0.55
    colors = ["#dc2626", "#f59e0b", "#2563eb", "#64748b", "#7c3aed"]

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Open cases by priority chart">']
    for g in range(5):
        gv = max_v * g / 4
        gy = pad_t + plot_h - (plot_h * gv / max_v)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" font-size="10" fill="#5b6370" text-anchor="end">{gv:.0f}</text>')
    for i, (cat, val) in enumerate(zip(categories, values)):
        bx = pad_l + slot * i + (slot - bar_w) / 2
        bh = plot_h * val / max_v
        by = pad_t + plot_h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{colors[i % len(colors)]}" rx="3"/>')
        parts.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" font-size="12" font-weight="700" fill="#1f2937" text-anchor="middle">{val}</text>')
        parts.append(f'<text x="{pad_l + slot * i + slot / 2:.1f}" y="{h - pad_b + 18}" font-size="10" fill="#5b6370" text-anchor="middle">{esc(cat)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# HTML SECTIONS
# --------------------------------------------------------------------------
def kpi_tiles_html(m):
    band_hex = BAND_HEX[m["band"]]
    tiles = [
        ("Mgmt Attention Score", f'{m["score"]} &middot; {m["band"]}', band_hex, "#fff"),
    ]
    tiles.append(("P1 Incidents", m["p1_open"], "#dc2626" if m["p1_open"] else "#fff", "#fff" if m["p1_open"] else "#1f2937"))
    tiles.append(("Aged High Priority (P2)", m["aged_p2"], "#f59e0b" if m["aged_p2"] else "#fff", "#fff" if m["aged_p2"] else "#1f2937"))
    tiles.append(("Backlog (Open Cases)", m["backlog"], "#fff", "#1f2937"))

    cells = []
    for label, value, bg, fg in tiles:
        border = "border:1px solid #e6e6e6;" if bg == "#fff" else ""
        cells.append(
            f'<div class="kpi" style="background:{bg};color:{fg};{border}">'
            f'<div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{value}</div></div>'
        )
    return '<div class="kpi-row">' + "".join(cells) + "</div>"


def risk_matrix_html(m):
    labels = CONFIG["age_bucket_labels"]
    head = "".join(f"<th>{esc(l)}</th>" for l in labels)
    rows = []
    for p in ["P1", "P2", "P3", "P4"]:
        cells = [f"<td class='rowhead'>{p}</td>"]
        for c_i in range(4):
            rag = CONFIG["risk_matrix"][p][c_i]
            cnt = m["matrix_counts"][p][c_i]
            txt = str(cnt) if cnt else "-"
            fg = "#fff"
            cells.append(f"<td style='background:{RAG_HEX[rag]};color:{fg};font-weight:700;text-align:center'>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="matrix"><thead><tr><th></th>' + head + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def watchlist_html(m):
    rows = []
    top5 = m["top5"]
    for i in range(len(top5)):
        r = top5.iloc[i]
        rag = r["RAG"]
        subj = esc(str(r["Subject"]))
        esc_flag = "Y" if r["State"] == "Escalation" else "N"
        rows.append(
            "<tr>"
            f"<td>{esc(r['Number'])}</td>"
            f"<td>{esc(r['Priority'])}</td>"
            f"<td style='text-align:center'>{r['Days Open']:.0f}</td>"
            f"<td style='text-align:center'><span class='pill' style='background:{RAG_HEX[rag]}'>{RAG_LABEL[rag][0]}</span></td>"
            f"<td>{subj}</td>"
            f"<td style='text-align:center'>{esc_flag}</td>"
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr>'
        "<th>Case</th><th>Priority</th><th>Days</th><th>RAG</th><th>Issue Summary</th><th>Escalated?</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def rca_table_html(m):
    rca = m["rca_cases"]
    rows = []
    for i in range(len(rca)):
        r = rca.iloc[i]
        rag = r["RAG"]
        subj = esc(str(r["Subject"]))
        esc_flag = "Y" if r["State"] == "Escalation" else "N"
        rows.append(
            "<tr>"
            f"<td>{esc(r['Number'])}</td>"
            f"<td>{esc(r['Priority'])}</td>"
            f"<td style='text-align:center'>{r['Days Open']:.0f}</td>"
            f"<td style='text-align:center'><span class='pill' style='background:{RAG_HEX[rag]}'>{RAG_LABEL[rag][0]}</span></td>"
            f"<td>{subj}</td>"
            f"<td style='text-align:center'>{esc_flag}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='6'>No Pending RCA cases</td></tr>")
    return (
        '<table class="data"><thead><tr>'
        "<th>Case</th><th>Priority</th><th>Days</th><th>RAG</th><th>Issue Summary</th><th>Escalated?</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def themes_html(m):
    rows = []
    for t in m["themes"]:
        open_txt = str(t["open_count"])
        if t["open_cases"]:
            open_txt += f' ({esc(", ".join(t["open_cases"][:3]))})'
        concern = ("Currently open &ndash; needs root-cause follow up" if t["open_count"]
                   else "Recurring historically &ndash; verify preventative fix is effective")
        rows.append(
            "<tr>"
            f"<td>{esc(t['theme'])}</td>"
            f"<td style='text-align:center'>{t['occurrences']}</td>"
            f"<td>{open_txt}</td>"
            f"<td>{concern}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4'>No repeat themes detected above threshold</td></tr>")
    return (
        '<table class="data"><thead><tr>'
        "<th>Theme (auto-detected)</th><th>Historical Occurrences</th><th>Currently Open</th><th>Service Concern</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def scoring_html(m):
    sw = CONFIG["score_weights"]
    caps = CONFIG["score_caps"]
    buckets = " / ".join(CONFIG["age_bucket_labels"])
    model = [
        "Points per open case, by age:",
        f"Age: {esc(buckets)}",
        f'P1 Critical: {" / ".join(str(x) for x in sw["P1"])}',
        f'P2 High: {" / ".join(str(x) for x in sw["P2"])}',
        f'P3 Moderate: {" / ".join(str(x) for x in sw["P3"])} (cap {caps["P3"]})',
        f'P4 Low: {" / ".join(str(x) for x in sw["P4"])} (cap {caps["P4"]})',
        "Score = 100 &minus; P1 &minus; P2 &minus; P3 &minus; P4",
    ]
    model_html = "<br>".join(model)
    drivers_html = "<br>".join(f"&bull; {esc(d)}" for d in m["drivers"])
    drivers_html += f'<br><strong>Score: {CONFIG["attention_start"]} &minus; {m["total_deduction"]} = {m["score"]}</strong>'
    band_hex = BAND_HEX[m["band"]]
    return f"""
    <div class="score-panel">
      <div class="score-circle" style="background:{band_hex}">
        <div class="score-num">{m['score']}</div>
        <div class="score-band">{m['band']}</div>
      </div>
      <div class="score-col">
        <div class="score-h">Scoring Model</div>
        <div class="score-body">{model_html}</div>
      </div>
      <div class="score-col">
        <div class="score-h">Today's Drivers</div>
        <div class="score-body">{drivers_html}</div>
      </div>
    </div>
    """


def appendix_html(m):
    labels = CONFIG["age_bucket_labels"]
    sw = CONFIG["score_weights"]
    caps = CONFIG["score_caps"]
    names = {"P1": "P1 Critical", "P2": "P2 High", "P3": "P3 Moderate", "P4": "P4 Low"}

    # ---- Weight rulebook table ----
    head = "".join(f"<th>{esc(l)}</th>" for l in labels)
    wrows = []
    for p in ["P1", "P2", "P3", "P4"]:
        cells = "".join(f"<td style='text-align:center'>{v}</td>" for v in sw[p])
        cap = caps[p]
        cap_txt = "none" if cap is None else str(cap)
        wrows.append(f"<tr><td class='rowhead'>{names[p]}</td>{cells}<td style='text-align:center'>{cap_txt}</td></tr>")
    weights_table = (
        "<table class='data'><thead><tr><th>Priority</th>" + head
        + "<th>Cap</th></tr></thead><tbody>" + "".join(wrows) + "</tbody></table>"
    )

    # ---- Worked calculation table ----
    crows = []
    for p in ["P1", "P2", "P3", "P4"]:
        b = m["score_breakdown"][p]
        counts, weights = b["counts"], b["weights"]
        products = " + ".join(f"{c}&times;{w}" for c, w in zip(counts, weights))
        cap = b["cap"]
        cap_txt = "none" if cap is None else str(cap)
        cap_flag = " (capped)" if cap is not None and b["raw"] > cap else ""
        crows.append(
            "<tr>"
            f"<td class='rowhead'>{names[p]}</td>"
            f"<td>{products}</td>"
            f"<td style='text-align:center'>{b['raw']}</td>"
            f"<td style='text-align:center'>{cap_txt}</td>"
            f"<td style='text-align:center'><strong>&minus;{b['applied']}{cap_flag}</strong></td>"
            "</tr>"
        )
    calc_table = (
        "<table class='data'><thead><tr>"
        "<th>Priority</th><th>Weighted (count &times; points)</th><th>Subtotal</th><th>Cap</th><th>Applied</th>"
        "</tr></thead><tbody>" + "".join(crows)
        + f"<tr><td colspan='4' style='text-align:right'><strong>Total deduction</strong></td>"
          f"<td style='text-align:center'><strong>&minus;{m['total_deduction']}</strong></td></tr>"
        + "</tbody></table>"
    )

    band_hex = BAND_HEX[m["band"]]
    bands = [("Green", "80 - 100", "#16a34a"), ("Amber", "50 - 79", "#f59e0b"), ("Red", "&lt; 50", "#dc2626")]
    bands_rows = "".join(
        f"<tr><td><span class='pill' style='background:{c}'>{name}</span></td><td>{rng}</td></tr>"
        for name, rng, c in bands
    )
    bands_table = "<table class='data'><thead><tr><th>Status</th><th>Score range</th></tr></thead><tbody>" + bands_rows + "</tbody></table>"

    return f"""
      <div class="appendix-grid">
        <div>
          <div class="section-title">A. Scoring rulebook &ndash; points deducted per open case</div>
          {weights_table}
          <div class="chart-note">Age buckets are days since the case was opened. P1/P2 have no cap; P3 and P4 contributions are each capped so a large low-priority backlog cannot dominate the score.</div>
          <div class="section-title" style="margin-top:18px">C. RAG bands</div>
          {bands_table}
        </div>
        <div>
          <div class="section-title">B. Today's worked calculation</div>
          {calc_table}
          <div class="final-calc">
            Score = 100 &minus; {m['total_deduction']} =
            <span class="final-score" style="background:{band_hex}">{m['score']} &middot; {m['band']}</span>
          </div>
          <div class="chart-note" style="margin-top:14px">
            <strong>Definitions:</strong> "Open" = any case not in state Closed / Resolved / Cancelled.
            Priority mapping: Critical&rarr;P1, High&rarr;P2, Moderate&rarr;P3, Low&rarr;P4.
            Repeat Incident Themes are auto-detected from recurring Subject patterns and are informational (they do not affect this score).
          </div>
        </div>
      </div>
    """


def build_html(m):
    asof_str = m["asof"].strftime("%d %b %Y")
    line = svg_line_chart(m["week_labels"], m["mttr_p1"], m["mttr_p2"], CONFIG["mttr_outlier_cap_days"])
    bars = svg_bar_chart(
        ["P1", "P2 High", "P3 Moderate", "P4 Low", "RCA Pending"],
        [m["open_by_priority"]["P1"], m["open_by_priority"]["P2"], m["open_by_priority"]["P3"], m["open_by_priority"]["P4"], m["rca_pending"]],
    )
    driver_strip = f'{m["band"]} &ndash; Drivers: ' + " | ".join(esc(d) for d in m["drivers"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BNY Managed Data Services - Executive Service Health</title>
<style>
  :root {{
    --navy:#13294b; --amber:#f59e0b; --green:#16a34a; --red:#dc2626;
    --bg:#f1f5f9; --surface:#fff; --border:#e6e6e6; --text:#1f2937; --muted:#5b6370;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',-apple-system,Roboto,sans-serif; background:var(--bg); color:var(--text); }}
  .page {{ max-width:1180px; margin:24px auto; background:var(--surface); box-shadow:0 4px 12px rgba(0,0,0,.08); }}
  .page + .page {{ margin-top:32px; }}
  .header {{ background:var(--navy); color:#fff; padding:16px 28px; }}
  .header h1 {{ font-size:1.35rem; font-weight:700; }}
  .header .sub {{ font-size:.8rem; color:#c7d2e0; margin-top:2px; }}
  .body {{ padding:20px 28px 28px; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:24px; }}
  .kpi {{ border-radius:8px; padding:12px 14px; min-height:78px; display:flex; flex-direction:column; justify-content:space-between; }}
  .kpi-label {{ font-size:.72rem; font-weight:700; opacity:.95; }}
  .kpi-value {{ font-size:1.5rem; font-weight:800; }}
  .grid2 {{ display:grid; grid-template-columns:340px 1fr; gap:24px; margin-bottom:22px; }}
  .section-title {{ font-size:.95rem; font-weight:700; margin-bottom:10px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.78rem; }}
  table.matrix td, table.matrix th {{ border:1px solid var(--border); padding:8px 6px; }}
  table.matrix th {{ background:#f5f5f5; font-weight:700; text-align:center; }}
  table.matrix .rowhead {{ font-weight:700; text-align:center; background:#fff; }}
  table.data th, table.data td {{ border:1px solid var(--border); padding:7px 9px; text-align:left; vertical-align:middle; }}
  table.data th {{ background:#f5f5f5; font-weight:700; }}
  .pill {{ display:inline-block; min-width:20px; padding:2px 7px; border-radius:9999px; color:#fff; font-weight:700; font-size:.72rem; }}
  .callout {{ margin:8px 0 10px; font-size:.9rem; font-weight:700; color:#b45309; }}
  .driver-strip {{ margin-top:14px; font-size:.78rem; color:var(--muted); padding:10px 12px; background:#f8fafc; border-left:3px solid var(--amber); border-radius:4px; }}
  .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:20px; }}
  .chart-card {{ border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
  .chart-note {{ font-size:.68rem; color:var(--muted); margin-top:4px; }}
  .score-panel {{ display:grid; grid-template-columns:180px 1fr 1fr; gap:20px; border:1px solid var(--border); border-radius:8px; padding:18px 20px; align-items:start; }}
  .score-circle {{ width:150px; height:150px; border-radius:50%; color:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
  .score-num {{ font-size:3rem; font-weight:800; line-height:1; }}
  .score-band {{ font-size:.9rem; font-weight:700; margin-top:4px; }}
  .score-h {{ font-size:.9rem; font-weight:700; margin-bottom:8px; }}
  .score-body {{ font-size:.8rem; line-height:1.7; color:var(--text); }}
  .band {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
  .swatch {{ width:16px; height:14px; border-radius:3px; display:inline-block; }}
  .footer-note {{ font-size:.68rem; color:var(--muted); padding:6px 28px 20px; }}
  .appendix-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:start; }}
  .final-calc {{ margin-top:16px; font-size:1rem; font-weight:700; }}
  .final-score {{ display:inline-block; color:#fff; padding:4px 14px; border-radius:8px; margin-left:6px; }}
  @media print {{
    body {{ background:#fff; }}
    .page {{ box-shadow:none; margin:0; max-width:none; page-break-after:always; }}
    .page:last-child {{ page-break-after:auto; }}
    @page {{ size:A4 landscape; margin:8mm; }}
  }}
</style>
</head>
<body>

  <!-- PAGE 1 -->
  <div class="page">
    <div class="header">
      <h1>BNY Managed Data Services &ndash; Executive Service Health</h1>
      <div class="sub">Point-in-time daily view | As at {asof_str}</div>
    </div>
    <div class="body">
      {kpi_tiles_html(m)}
      <div class="grid2">
        <div>
          <div class="section-title">Priority &times; Age Risk Matrix</div>
          {risk_matrix_html(m)}
        </div>
        <div>
          <div class="section-title">Executive Watchlist &ndash; Top 5</div>
          {watchlist_html(m)}
        </div>
      </div>
      <div class="callout">&#9888; Repeat Incident Themes (systemic patterns requiring root-cause action, not just ticket closure)</div>
      {themes_html(m)}
      <div class="driver-strip">{driver_strip}</div>
    </div>
  </div>

  <!-- PAGE 2 -->
  <div class="page">
    <div class="header">
      <h1>BNY Managed Data Services &ndash; Trend, Backlog &amp; Attention Score</h1>
      <div class="sub">As at {asof_str}</div>
    </div>
    <div class="body">
      <div class="charts">
        <div class="chart-card">
          <div class="section-title">Mean Time to Resolution &ndash; P1 &amp; P2 (last {CONFIG["mttr_months"]} months, days)</div>
          {line}
          <div class="chart-note">Monthly median (4-week blocks); capped at {CONFIG["mttr_outlier_cap_days"]}d so a single legacy backlog closure does not distort the trend.</div>
        </div>
        <div class="chart-card">
          <div class="section-title">Open Cases by Priority (as at {asof_str})</div>
          {bars}
          <div class="chart-note">RCA Pending (purple) are open cases parked awaiting root-cause analysis &ndash; excluded from the other priority bars and the score.</div>
        </div>
      </div>
      <div class="section-title">Management Attention Score</div>
      {scoring_html(m)}
      <div class="section-title" style="margin-top:22px">Pending RCA &ndash; Root Cause Analysis in Progress ({m['rca_pending']})</div>
      <div class="chart-note" style="margin-bottom:8px">These open cases are parked awaiting root-cause analysis and are excluded from the Management Attention Score, KPIs, risk matrix and Executive Watchlist.</div>
      {rca_table_html(m)}
    </div>
    <div class="footer-note">
      Repeat Incident Themes are auto-detected from recurring Subject text. The Management Attention Score is derived solely from open case priority and age (see Appendix). Review before external distribution.
    </div>
  </div>

  <!-- PAGE 3 - APPENDIX -->
  <div class="page">
    <div class="header">
      <h1>Appendix &ndash; Management Attention Score Methodology</h1>
      <div class="sub">Full transparency of how today's score ({m['score']} &middot; {m['band']}) is calculated | As at {asof_str}</div>
    </div>
    <div class="body">
      {appendix_html(m)}
    </div>
  </div>

</body>
</html>
"""


def main():
    source_path = find_latest_source_file(CONFIG["source_dir"], CONFIG["source_glob"])
    asof = pd.Timestamp(parse_asof_from_filename(source_path))
    print(f"Using source file: {source_path}")
    print(f"As-of timestamp:   {asof}")

    df = load_cases(source_path)
    m = compute_metrics(df, asof)

    html = build_html(m)
    out_name = f'BNY_Executive_Dashboard_Services_{asof.strftime("%d%m%Y")}.html'
    out_path = os.path.join(CONFIG["source_dir"], out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Saved: {out_path}")

    print("\n--- Summary ---")
    print(f"Mgmt Attention Score: {m['score']} ({m['band']})")
    print(f"P1 open: {m['p1_open']} | Aged P2: {m['aged_p2']} | Repeat incidents: {m['repeat_incidents']}")
    print(f"Backlog: {m['backlog']}")
    print("Score drivers:", "; ".join(m["drivers"]))


if __name__ == "__main__":
    main()
