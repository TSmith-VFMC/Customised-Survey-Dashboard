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

import json
import os
from urllib.parse import quote

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

# Same OneDrive-synced SharePoint folder publish_to_sharepoint.py copies the
# PDF and source xlsx into - used to build the Appendix B hyperlink.
SHAREPOINT_FOLDER_URL = (
    "https://vfmcorp.sharepoint.com/sites/External-VFMCBNYMellon"
    "/Shared%20Documents/General/Operations/VFMC%20Daily%20Scorecard/"
)

# Rolling score history, written to the synced folder so the trend persists
# across runs and users.
HISTORY_FILE = "dashboard_history.json"

# Where the "Ask AI" button sends the user. Microsoft 365 (enterprise) Copilot,
# so BNY data stays inside the approved tenant. Change if your org uses a
# different Copilot entry point.
COPILOT_URL = "https://m365.cloud.microsoft/chat"


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
    # x labels (may contain a second line, e.g. "Month 1\n1 Jul-28 Jul")
    n_lbl = len(week_labels)
    for i, lbl in enumerate(week_labels):
        lines = str(lbl).split("\n")
        xi = x(i)
        # Keep edge labels inside the viewBox so the date range isn't clipped.
        if i == 0:
            anchor = "start"
        elif i == n_lbl - 1:
            anchor = "end"
        else:
            anchor = "middle"
        for li, line in enumerate(lines):
            weight = "600" if li == 0 else "400"
            fill = "#1f2937" if li == 0 else "#8a929e"
            parts.append(
                f'<text x="{xi:.1f}" y="{h - pad_b + 16 + li * 12}" font-size="10" '
                f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(line)}</text>'
            )

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
    # score and P1 span the full row; aged P2 and backlog share the last row
    tiles = [
        ("Mgmt Attention Score", f'{m["score"]} &middot; {m["band"]}', band_hex, "#fff", True),
        ("P1 Incidents", m["p1_open"], "#dc2626" if m["p1_open"] else "#fff", "#fff" if m["p1_open"] else "#1f2937", True),
        ("Aged High Priority (P2)", m["aged_p2"], "#f59e0b" if m["aged_p2"] else "#fff", "#fff" if m["aged_p2"] else "#1f2937", False),
        ("Backlog (Open Cases)", m["backlog"], "#fff", "#1f2937", False),
    ]

    cells = []
    for label, value, bg, fg, full_span in tiles:
        border = "border:1px solid #e6e6e6;" if bg == "#fff" else ""
        span = "grid-column:1 / -1;" if full_span else ""
        cells.append(
            f'<div class="kpi" style="background:{bg};color:{fg};{border}{span}">'
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
        case_type = esc(str(r.get("Case Type", "") or ""))
        state = esc(str(r.get("State", "") or ""))
        submitted_by = esc(str(r.get("Submitted By", "") or ""))
        assigned_to = esc(str(r.get("Assigned To", "") or ""))
        esc_flag = "Y" if r["State"] == "Escalation" else "N"
        rows.append(
            "<tr>"
            f"<td>{esc(r['Number'])}</td>"
            f"<td>{esc(r['Priority'])}</td>"
            f"<td style='text-align:center'>{r['Days Open']:.0f}</td>"
            f"<td style='text-align:center'><span class='pill' style='background:{RAG_HEX[rag]}'>{r['PriorityCode']}</span></td>"
            f"<td>{subj}</td>"
            f"<td>{case_type}</td>"
            f"<td>{state}</td>"
            f"<td>{submitted_by}</td>"
            f"<td>{assigned_to}</td>"
            f"<td style='text-align:center'>{esc_flag}</td>"
            "</tr>"
        )
    return (
        '<table class="data watchlist"><thead><tr>'
        "<th>Case</th><th>Priority</th><th>Days</th><th>RAG</th><th>Issue Summary</th><th>Case Type</th><th>State</th><th>Submitted By</th><th>Assigned To</th><th>Esc?</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def rca_table_html(m):
    rca = m["rca_cases"]
    rows = []
    for i in range(len(rca)):
        r = rca.iloc[i]
        rag = r["RAG"]
        subj = esc(str(r["Subject"]))
        case_type = esc(str(r.get("Case Type", "") or ""))
        esc_flag = "Y" if r["State"] == "Escalation" else "N"
        rows.append(
            "<tr>"
            f"<td>{esc(r['Number'])}</td>"
            f"<td>{esc(r['Priority'])}</td>"
            f"<td style='text-align:center'>{r['Days Open']:.0f}</td>"
            f"<td style='text-align:center'><span class='pill' style='background:{RAG_HEX[rag]}'>{r['PriorityCode']}</span></td>"
            f"<td>{subj}</td>"
            f"<td>{case_type}</td>"
            f"<td style='text-align:center'>{esc_flag}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='7'>No Pending RCA cases</td></tr>")
    return (
        '<table class="data"><thead><tr>'
        "<th>Case</th><th>Priority</th><th>Days</th><th>RAG</th><th>Issue Summary</th><th>Case Type</th><th>Escalated?</th>"
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
    driver_items = "".join(f'<div class="driver-item">{esc(d)}</div>' for d in m["drivers"])
    driver_items += (
        f'<div class="driver-item total">Score: {CONFIG["attention_start"]} '
        f'&minus; {m["total_deduction"]} = {m["score"]} &middot; {m["band"]}</div>'
    )
    return f"""
    <div class="score-panel">
      <div class="score-h">Today's Drivers</div>
      <div class="drivers-row">{driver_items}</div>
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
            Repeat Incident Themes are derived by text-pattern matching on the Subject field (recurring identifiers/phrases appearing 3+ times) &ndash; not a case category field &ndash; and are informational (they do not affect this score).
            Cases with State = Awaiting Info score input = 0 (individually) since they are pending client-provided information, not team action &ndash; they still appear as open cases in the KPIs, Risk Matrix and Watchlist.
          </div>
        </div>
      </div>
    """


def _history_path():
    return os.path.join(CONFIG["source_dir"], HISTORY_FILE)


def _conflict_copies():
    """OneDrive conflict copies of the history file, e.g.
    'dashboard_history-CPC-tsmit-KCWU8.json', created when more than one
    machine/user writes the shared file. Returns their full paths."""
    import glob

    stem = HISTORY_FILE[: -len(".json")]
    pattern = os.path.join(CONFIG["source_dir"], f"{stem}-*.json")
    return sorted(glob.glob(pattern))


def _read_history_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _merge_snapshots(*snapshot_lists):
    """Merge snapshot lists by date (one entry per day), newest kept last.
    Earlier lists win on a date clash, so pass the authoritative file first."""
    by_date = {}
    for snaps in snapshot_lists:
        for entry in snaps:
            date = entry.get("date")
            if date and date not in by_date:
                by_date[date] = entry
    merged = sorted(by_date.values(), key=lambda h: h.get("date", ""))
    return merged[-60:]


def load_history():
    """Return recorded run snapshots (list of dicts), oldest first. Also folds
    in any OneDrive conflict copies so the trend is complete even before the
    next write cleans them up."""
    main = _read_history_file(_history_path())
    copies = [_read_history_file(p) for p in _conflict_copies()]
    return _merge_snapshots(main, *copies)


def record_history(m, asof):
    """Append (or replace) today's score snapshot, keyed by as-of date so a
    re-run on the same day updates one entry instead of duplicating. Merges in
    any OneDrive conflict copies, writes the combined file, then deletes the
    stray copies so they stop accumulating. Keeps the most recent 60 runs."""
    date_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
    entry = {
        "date": date_str,
        "score": int(m["score"]),
        "backlog": int(m["backlog"]),
        "p1": int(m["p1_open"]),
        "agedP2": int(m["aged_p2"]),
    }
    copies = _conflict_copies()
    merged = _merge_snapshots(
        [entry],
        _read_history_file(_history_path()),
        *[_read_history_file(p) for p in copies],
    )
    try:
        with open(_history_path(), "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
    except OSError:
        return merged
    for path in copies:
        try:
            os.remove(path)
        except OSError:
            pass
    return merged


def svg_sparkline(scores, dot_color, width=210, height=46):
    if len(scores) < 2:
        return ""
    pts = scores[-14:]
    n = len(pts)
    pad = 5
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1

    def x(i):
        return pad + (width - 2 * pad) * i / (n - 1)

    def y(v):
        return pad + (height - 2 * pad) * (1 - (v - lo) / rng)

    line_pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(pts))
    lx, ly = x(n - 1), y(pts[-1])
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-label="Attention score trend sparkline">'
        f'<polyline points="{line_pts}" fill="none" stroke="#13294b" stroke-width="2"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{dot_color}"/>'
        "</svg>"
    )


def trend_html(m, history):
    scores = [h["score"] for h in history if isinstance(h.get("score"), (int, float))]
    band_hex = BAND_HEX[m["band"]]
    spark = svg_sparkline(scores, band_hex)
    if len(scores) >= 2:
        prev, cur = scores[-2], scores[-1]
        diff = cur - prev
        if diff > 0:
            arrow, col, txt = "&#9650;", "#16a34a", f"up {diff} from {prev}"
        elif diff < 0:
            arrow, col, txt = "&#9660;", "#dc2626", f"down {abs(diff)} from {prev}"
        else:
            arrow, col, txt = "&#8212;", "#5b6370", f"unchanged from {prev}"
        delta = f'<span class="trend-delta" style="color:{col}">{arrow} {txt}</span> last run'
    elif len(scores) == 1:
        delta = '<span class="trend-note">First recorded run &ndash; the trend line builds from here.</span>'
    else:
        delta = '<span class="trend-note">No history recorded yet.</span>'
    spark_html = f'<div class="trend-spark">{spark}</div>' if spark else ""
    runs_txt = f'{len(scores)} run(s) recorded' if scores else ""
    return (
        '<div class="trend-block">'
        f'<div class="trend-score" style="color:{band_hex}">{m["score"]}</div>'
        '<div class="trend-meta">'
        f'<div class="trend-line1">Management Attention Score &middot; {m["band"]}</div>'
        f'<div class="trend-line2">{delta}</div>'
        f'<div class="trend-line3">{runs_txt}</div>'
        '</div>'
        f'{spark_html}'
        "</div>"
    )


# Screen-only search / sort / filter controls for the Appendix B open-case table.
OB_TOOLBAR = (
    '<div class="ob-toolbar">'
    '<input type="text" id="obSearch" placeholder="Search case, subject, person\u2026" '
    'oninput="obApplyFilters()" aria-label="Search open cases">'
    '<select id="obRag" onchange="obApplyFilters()" aria-label="Filter by RAG">'
    '<option value="">All RAG</option><option value="R">Red</option>'
    '<option value="A">Amber</option><option value="G">Green</option></select>'
    '<select id="obPriority" onchange="obApplyFilters()" aria-label="Filter by priority">'
    '<option value="">All priorities</option><option value="P1">P1</option>'
    '<option value="P2">P2</option><option value="P3">P3</option>'
    '<option value="P4">P4</option></select>'
    '<span class="ob-count" id="obCount"></span>'
    '</div>'
)

OPEN_CASES_SCRIPT = """
<script>
(function(){
  var table = document.getElementById('openCasesTable');
  if(!table || !table.tBodies.length) return;
  var tbody = table.tBodies[0];
  var allRows = Array.prototype.slice.call(tbody.rows);
  function byId(id){ return document.getElementById(id); }
  window.obApplyFilters = function(){
    var s = byId('obSearch'), r = byId('obRag'), p = byId('obPriority');
    var q = (s && s.value || '').toLowerCase();
    var rag = r && r.value || '';
    var pri = p && p.value || '';
    var shown = 0;
    allRows.forEach(function(row){
      if(row.getAttribute('data-empty')) return;
      var ok = (!q || row.textContent.toLowerCase().indexOf(q) !== -1)
        && (!rag || row.getAttribute('data-rag') === rag)
        && (!pri || row.getAttribute('data-priority') === pri);
      row.style.display = ok ? '' : 'none';
      if(ok) shown++;
    });
    var c = byId('obCount');
    if(c) c.textContent = shown + ' of ' + allRows.length + ' shown';
  };
  var headers = table.tHead ? table.tHead.rows[0].cells : [];
  var current = {col:-1, dir:''};
  function val(row, idx){ var c = row.cells[idx]; return c ? c.textContent.trim() : ''; }
  Array.prototype.forEach.call(headers, function(th, idx){
    th.classList.add('sortable');
    th.addEventListener('click', function(){
      var dir = (current.col === idx && current.dir === 'asc') ? 'desc' : 'asc';
      current = {col: idx, dir: dir};
      var rows = allRows.filter(function(r){ return !r.getAttribute('data-empty'); });
      rows.sort(function(a, b){
        var va = val(a, idx), vb = val(b, idx);
        var na = parseFloat(va), nb = parseFloat(vb);
        var cmp = (!isNaN(na) && !isNaN(nb)) ? (na - nb) : va.localeCompare(vb);
        return dir === 'asc' ? cmp : -cmp;
      });
      rows.forEach(function(r){ tbody.appendChild(r); });
      Array.prototype.forEach.call(headers, function(h){ h.removeAttribute('data-sort'); });
      th.setAttribute('data-sort', dir);
    });
  });
  window.obApplyFilters();
})();
</script>
"""


def _ai_context(m, xlsx_url, pdf_url):
    """Compact JSON knowledge payload the in-page assistant hands to Copilot:
    the headline figures, watchlist, themes, MTTR trend, the full open-case
    list, and links to the Excel source + PDF report."""
    def case_row(r):
        return {
            "case": str(r["Number"]),
            "priority": str(r["Priority"]),
            "daysOpen": int(round(float(r["Days Open"]))),
            "subject": str(r["Subject"]),
            "state": "" if pd.isna(r.get("State", "")) else str(r.get("State", "")),
            "assignedTo": "" if pd.isna(r.get("Assigned To", "")) else str(r.get("Assigned To", "")),
            "pendingRca": bool(r.get("PendingRCA", False)),
        }

    open_df = m["open_df"]
    top5 = m["top5"]
    return {
        "asOf": pd.Timestamp(m["asof"]).strftime("%Y-%m-%d"),
        "managementAttentionScore": int(m["score"]),
        "band": m["band"],
        "backlogOpenCases": int(m["backlog"]),
        "p1Open": int(m["p1_open"]),
        "agedP2": int(m["aged_p2"]),
        "repeatIncidents": int(m["repeat_incidents"]),
        "rcaPendingExcluded": int(m["rca_pending"]),
        "openByPriority": m["open_by_priority"],
        "scoreDrivers": m["drivers"],
        "mttrTrend": {
            "periods": [w.replace("\n", " ") for w in m["week_labels"]],
            "p1Days": m["mttr_p1"],
            "p2Days": m["mttr_p2"],
        },
        "executiveWatchlistTop5": [case_row(top5.iloc[i]) for i in range(len(top5))],
        "repeatThemes": [
            {"theme": t["theme"], "historicalOccurrences": t["occurrences"], "openNow": t["open_count"]}
            for t in m["themes"]
        ],
        "openCases": [case_row(open_df.iloc[i]) for i in range(len(open_df))],
        "links": {"excelSourceData": xlsx_url, "pdfReport": pdf_url},
    }


# Floating "Ask AI" assistant (screen only). Packages the dashboard data +
# the user's question into a prompt, copies it, and opens M365 Copilot - no
# data leaves the approved tenant and no API key is embedded in the file.
# __CTX__ and __COPILOT_URL__ are substituted in build_html.
AI_ASSISTANT_SCRIPT = """
<script>
(function(){
  window.BNY_CONTEXT = __CTX__;
  var COPILOT_URL = "__COPILOT_URL__";
  var c = window.BNY_CONTEXT;
  var PRESETS = [
    "Summarise today's service health in 3 short bullets for an executive.",
    "Explain why the Management Attention Score is " + c.managementAttentionScore + " today, using the score drivers.",
    "Which open cases are the biggest risks right now, and why? List the top 3.",
    "Draft a brief, professional status update email to the client about today's position.",
    "What are the repeat incident themes and what root-cause actions do they suggest?"
  ];
  function preamble(){
    return "You are a service-delivery analyst assistant for the BNY Managed Data Services daily dashboard.\\n"
      + "STRICT GROUNDING RULES - follow exactly:\\n"
      + "1. Answer using ONLY the DASHBOARD DATA (JSON) provided below, which is current as at " + c.asOf + ".\\n"
      + "2. Do NOT use any outside, general, web, training or other organisational knowledge, and do NOT guess or infer beyond what the data states.\\n"
      + "3. If the answer cannot be found in the data below, reply exactly: \\\"That information isn't in today's dashboard data.\\\"\\n"
      + "4. You may point the reader to the linked Excel export (in the data) for row-level detail, but every statement you make must be supported by the data below.\\n"
      + "Be concise and executive-ready.\\n\\n"
      + "DASHBOARD DATA (JSON):\\n" + JSON.stringify(c, null, 2) + "\\n\\n";
  }
  function buildPrompt(q){ return preamble() + "QUESTION:\\n" + q; }
  function byId(id){ return document.getElementById(id); }
  function toast(msg){
    var t = byId('aiToast');
    if(!t) return;
    if(msg) t.textContent = msg;
    t.classList.add('show');
    setTimeout(function(){ t.classList.remove('show'); }, 3200);
  }
  // Legacy copy path - works where the async Clipboard API is blocked by a
  // document permissions policy (common in locked-down / embedded browsers).
  function fallbackCopy(txt){
    try {
      var ta = document.createElement('textarea');
      ta.value = txt;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed'; ta.style.top = '0'; ta.style.left = '0';
      ta.style.width = '1px'; ta.style.height = '1px'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch(e){ ok = false; }
      document.body.removeChild(ta);
      return ok;
    } catch(e){ return false; }
  }
  // Resolves true if the text reached the clipboard, false otherwise. Never
  // rejects, so a blocked Clipboard API can't surface as an unhandled rejection.
  function copyText(txt){
    return new Promise(function(resolve){
      if(navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText(txt).then(
          function(){ resolve(true); },
          function(){ resolve(fallbackCopy(txt)); }
        );
      } else {
        resolve(fallbackCopy(txt));
      }
    });
  }
  function showManual(promptText){
    var out = byId('aiPromptOut'), box = byId('aiFallback');
    if(out){ out.value = promptText; }
    if(box){ box.style.display = 'block'; }
    if(out){ out.focus(); out.select(); }
  }
  function hideManual(){
    var box = byId('aiFallback');
    if(box){ box.style.display = 'none'; }
  }
  function showStatus(html){
    var s = byId('aiStatus');
    if(!s) return;
    s.innerHTML = html;
    s.style.display = 'block';
  }
  function hideStatus(){
    var s = byId('aiStatus');
    if(s){ s.style.display = 'none'; }
  }
  function currentQuestion(){
    var el = byId('aiQuestion');
    return (el && el.value || '').trim();
  }
  // ---- In-page answer engine: replies using ONLY window.BNY_CONTEXT ----
  function esc(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function caseLine(x){
    var who = x.assignedTo ? (' \\u2013 ' + esc(x.assignedTo)) : '';
    var st = x.state ? (', ' + esc(x.state)) : '';
    return '<strong>' + esc(x.case) + '</strong> (' + esc(x.priority) + ', ' + x.daysOpen + 'd' + st + who + '): ' + esc(x.subject);
  }
  function ansSummary(){
    var ob = c.openByPriority || {};
    var top = (c.executiveWatchlistTop5 || [])[0];
    var h = '<ul>';
    h += '<li><strong>Management Attention Score ' + c.managementAttentionScore + ' (' + esc(c.band) + ')</strong> as at ' + esc(c.asOf) + ' \\u2013 ' + (100 - c.managementAttentionScore) + ' points deducted from 100.</li>';
    h += '<li><strong>' + c.backlogOpenCases + ' open cases</strong> (P1 ' + (ob.P1||0) + ', P2 ' + (ob.P2||0) + ', P3 ' + (ob.P3||0) + ', P4 ' + (ob.P4||0) + '); ' + c.p1Open + ' P1 open, ' + c.agedP2 + ' aged P2 over 30 days.</li>';
    if(top){ h += '<li><strong>Top risk:</strong> ' + caseLine(top) + '</li>'; }
    h += '</ul>';
    return h;
  }
  function ansScore(){
    var h = '<p>The Management Attention Score is <strong>' + c.managementAttentionScore + '</strong> (' + esc(c.band) + ') \\u2013 100 minus ' + (100 - c.managementAttentionScore) + ' points of deductions from open cases:</p><ul>';
    (c.scoreDrivers || []).forEach(function(d){ h += '<li>' + esc(d) + '</li>'; });
    h += '</ul>';
    return h;
  }
  function ansRisks(n){
    var list = (c.executiveWatchlistTop5 || []).slice(0, n || 3);
    if(!list.length){ return '<p>No watchlist items in today\\u2019s data.</p>'; }
    var h = '<p>Top ' + list.length + ' risks right now (priority, then age):</p><ol>';
    list.forEach(function(x){ h += '<li>' + caseLine(x) + '</li>'; });
    h += '</ol>';
    return h;
  }
  function ansEmail(){
    var top = (c.executiveWatchlistTop5 || []).slice(0, 3);
    var lines = top.map(function(x){ return '- ' + x.case + ' (' + x.priority + ', ' + x.daysOpen + 'd): ' + x.subject; }).join('\\n');
    var body = 'Subject: BNY Managed Data Services \\u2013 Daily Service Update (' + c.asOf + ')\\n\\n'
      + 'Hi all,\\n\\n'
      + 'Today\\u2019s service position: Management Attention Score ' + c.managementAttentionScore + ' (' + c.band + '). '
      + 'There are ' + c.backlogOpenCases + ' open cases (' + c.p1Open + ' P1, ' + c.agedP2 + ' aged P2 over 30 days).\\n\\n'
      + 'Key items we are focused on:\\n' + (lines || '- None') + '\\n\\n'
      + 'We will continue to prioritise the above and provide a further update tomorrow.\\n\\nRegards,';
    return '<p>Draft status email (editable):</p><textarea class="ai-draft" readonly>' + esc(body) + '</textarea>';
  }
  function ansThemes(){
    var th = c.repeatThemes || [];
    if(!th.length){ return '<p>No repeat incident themes in today\\u2019s data.</p>'; }
    var h = '<p>Repeat incident themes (historical occurrences / open now):</p><ul>';
    th.forEach(function(t){ h += '<li><strong>' + esc(t.theme) + '</strong> \\u2013 ' + t.historicalOccurrences + ' historical, ' + t.openNow + ' open now.</li>'; });
    h += '</ul><p class="ai-sub">Suggested focus: prioritise a root-cause review on the highest-recurrence themes and confirm permanent fixes are tracked to closure.</p>';
    return h;
  }
  function ansP1(){
    var all = (c.openCases || []).filter(function(x){ return /critical/i.test(x.priority); });
    var active = all.filter(function(x){ return !x.pendingRca; });
    var rca = all.filter(function(x){ return x.pendingRca; });
    var h = '<p>There ' + (c.p1Open === 1 ? 'is' : 'are') + ' <strong>' + c.p1Open + ' open P1</strong> (Critical) case' + (c.p1Open === 1 ? '' : 's') + ' in the attention score:</p><ul>';
    active.forEach(function(x){ h += '<li>' + caseLine(x) + '</li>'; });
    h += '</ul>';
    if(rca.length){
      h += '<p class="ai-sub">Plus ' + rca.length + ' P1 case' + (rca.length === 1 ? '' : 's') + ' parked pending RCA (excluded from the score):</p><ul>';
      rca.forEach(function(x){ h += '<li>' + caseLine(x) + '</li>'; });
      h += '</ul>';
    }
    return h;
  }
  function ansAgedP2(){
    var all = (c.openCases || []).filter(function(x){ return /high/i.test(x.priority) && x.daysOpen > 30; });
    var active = all.filter(function(x){ return !x.pendingRca; });
    var h = '<p>There ' + (c.agedP2 === 1 ? 'is' : 'are') + ' <strong>' + c.agedP2 + ' aged P2</strong> (High priority, open over 30 days):</p><ul>';
    active.forEach(function(x){ h += '<li>' + caseLine(x) + '</li>'; });
    h += '</ul>';
    return h;
  }
  function ansBacklog(){
    var ob = c.openByPriority || {};
    var h = '<p><strong>' + c.backlogOpenCases + ' open cases</strong> as at ' + esc(c.asOf) + ':</p><ul>'
      + '<li>P1 Critical: ' + (ob.P1||0) + '</li>'
      + '<li>P2 High: ' + (ob.P2||0) + '</li>'
      + '<li>P3 Moderate: ' + (ob.P3||0) + '</li>'
      + '<li>P4 Low: ' + (ob.P4||0) + '</li></ul>';
    if(c.rcaPendingExcluded){ h += '<p class="ai-sub">' + c.rcaPendingExcluded + ' further case' + (c.rcaPendingExcluded === 1 ? ' is' : 's are') + ' parked pending RCA and excluded from these figures.</p>'; }
    return h;
  }
  function ansMttr(){
    var t = c.mttrTrend || {}, per = t.periods || [], p1 = t.p1Days || [], p2 = t.p2Days || [];
    if(!per.length){ return '<p>No MTTR trend data in today\\u2019s data.</p>'; }
    var i = per.length - 1;
    return '<p>Median time-to-resolve (most recent period, ' + esc(per[i]) + '):</p><ul>'
      + '<li>P1 Critical: ' + p1[i] + ' days</li>'
      + '<li>P2 High: ' + p2[i] + ' days</li></ul>';
  }
  function answerLocally(q){
    var s = (q || '').toLowerCase();
    function has(){ for(var i=0;i<arguments.length;i++){ if(s.indexOf(arguments[i]) >= 0) return true; } return false; }
    if(has('email','status update','client update','draft')) return ansEmail();
    if(has('repeat','theme','recurr')) return ansThemes();
    if(has('aged') && has('p2','high')) return ansAgedP2();
    if(has('mttr','resolution','time to resolve','resolve time')) return ansMttr();
    if(has('score','why is','attention','driver')) return ansScore();
    if(has('risk','biggest','top 3','top three','worst','highest priorit')) return ansRisks(3);
    if(has('p1','critical')) return ansP1();
    if(has('backlog','how many open','total open','open cases','how many case','how many are open')) return ansBacklog();
    if(has('summar','health','overview','bullet','snapshot','position','how are we')) return ansSummary();
    return null;
  }
  function renderAnswer(q){
    var ans = byId('aiAnswer');
    if(!ans) return false;
    var html = answerLocally(q);
    if(html){
      ans.innerHTML = '<div class="ai-answer-q">' + esc(q) + '</div>' + html
        + '<div class="ai-answer-src">Answered from today\\u2019s dashboard data only (as at ' + esc(c.asOf) + ').</div>';
      ans.style.display = 'block';
      hideStatus();
      return true;
    }
    return false;
  }
  window.aiAnswer = function(){
    var q = currentQuestion();
    if(!q){ toast('Type a question or pick a standard question first'); return; }
    if(!renderAnswer(q)){
      var ans = byId('aiAnswer');
      if(ans){ ans.style.display = 'none'; }
      showStatus('I can answer the standard questions directly from the data. For open-ended analysis, use <strong>Ask Copilot</strong> to send this question together with the dashboard data.');
    }
  };
  window.aiToggle = function(){
    var p = byId('aiPanel');
    if(p) p.classList.toggle('open');
  };
  window.aiCopy = function(){
    var q = currentQuestion();
    if(!q){ toast('Type a question or pick a standard prompt first'); return; }
    var promptText = buildPrompt(q);
    copyText(promptText).then(function(ok){
      if(ok){
        hideManual();
        showStatus('<strong>&#10003; Prompt copied.</strong> Open Microsoft Copilot, click its message box, press <strong>Ctrl+V</strong> then <strong>Enter</strong>.');
        toast('Prompt copied');
      } else {
        hideStatus();
        showManual(promptText);
        toast('Couldn\\u2019t auto-copy \\u2013 select the text below and press Ctrl+C');
      }
    });
  };
  window.aiAsk = function(){
    var q = currentQuestion();
    if(!q){ toast('Type a question or pick a standard prompt first'); return; }
    var promptText = buildPrompt(q);
    copyText(promptText).then(function(ok){
      if(ok){
        hideManual();
        showStatus('<strong>&#10003; Prompt copied &amp; Copilot opened in a new tab.</strong> Switch to that tab, click the message box, press <strong>Ctrl+V</strong> then <strong>Enter</strong> to get your answer.');
        toast('Prompt copied \\u2013 now paste into Copilot (Ctrl+V)');
        window.open(COPILOT_URL, '_blank', 'noopener');
      } else {
        hideStatus();
        showManual(promptText);
        toast('Couldn\\u2019t auto-copy \\u2013 copy the text below, then click Open Copilot');
      }
    });
  };
  var box = byId('aiPresets');
  if(box){
    PRESETS.forEach(function(p){
      var b = document.createElement('button');
      b.className = 'ai-preset'; b.type = 'button'; b.textContent = p;
      b.addEventListener('click', function(){
        var el = byId('aiQuestion');
        if(el){ el.value = p; }
        renderAnswer(p);
      });
      box.appendChild(b);
    });
  }
  var link = byId('aiOpenLink');
  if(link){ link.setAttribute('href', COPILOT_URL); }
})();
</script>
"""


def appendix_b_html(m, xlsx_url, pdf_url):
    def field(r, col):
        val = r.get(col, "")
        return "" if pd.isna(val) else str(val)
    open_df = m["open_df"]
    rows = []
    for i in range(len(open_df)):
        r = open_df.iloc[i]
        rag = r["RAG"]
        subj = esc(str(r["Subject"]))
        state = esc(field(r, "State"))
        case_type = esc(field(r, "Case Type"))
        creator = esc(field(r, "Submitted By"))
        assignee = esc(field(r, "Assigned To"))
        environment = esc(field(r, "Environment"))
        component = esc(field(r, "Component"))
        esc_flag = "Y" if r["State"] == "Escalation" else "N"
        updated_dt = r.get("Updated_dt")
        updated_txt = updated_dt.strftime("%d %b %Y") if pd.notna(updated_dt) else esc(str(r.get("Updated", "") or ""))
        rows.append(
            f"<tr data-rag=\"{rag}\" data-priority=\"{esc(r['PriorityCode'])}\">"
            f"<td>{esc(r['Number'])}</td>"
            f"<td>{esc(r['Priority'])}</td>"
            f"<td style='text-align:center'>{r['Days Open']:.0f}</td>"
            f"<td style='text-align:center'><span class='pill' style='background:{RAG_HEX[rag]}'>{r['PriorityCode']}</span></td>"
            f"<td>{subj}</td>"
            f"<td>{state}</td>"
            f"<td style='text-align:center'>{esc_flag}</td>"
            f"<td>{case_type}</td>"
            f"<td>{creator}</td>"
            f"<td>{assignee}</td>"
            f"<td>{updated_txt}</td>"
            f"<td>{environment}</td>"
            f"<td>{component}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr data-empty='1'><td colspan='13'>No open cases</td></tr>")
    table = (
        '<table class="data open-cases" id="openCasesTable"><thead><tr>'
        "<th>Case</th><th>Priority</th><th>Days</th><th>RAG</th><th>Issue Summary</th><th>State</th>"
        "<th>Esc?</th><th>Case Type</th><th>Creator</th><th>Assignee</th><th>Update</th>"
        "<th>Environment</th><th>Component</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    return f"""
      <div class="chart-note" style="margin-bottom:10px">
        Downloads (archive):
        <a href="{pdf_url}" target="_blank" rel="noopener">PDF</a> &middot;
        <a href="{xlsx_url}" target="_blank" rel="noopener">source data ({esc(os.path.basename(xlsx_url))})</a>
      </div>
      <div class="section-title">All open cases ({len(open_df)}) &ndash; sorted by age (oldest first), then priority</div>
      {OB_TOOLBAR}
      {table}
      {OPEN_CASES_SCRIPT}
    """


def _freshness(asof):
    """Return (age_days, is_stale, badge_html, banner_html) comparing the export
    date to today, so an expired-session run that silently reused an old export
    can't pass unnoticed."""
    asof_date = pd.Timestamp(asof).normalize()
    today = pd.Timestamp.now().normalize()
    age_days = int((today - asof_date).days)
    asof_str = asof_date.strftime("%d %b %Y")
    if age_days <= 0:
        badge = f'<span class="fresh-badge fresh-ok">&#10003; Data current &middot; {asof_str}</span>'
        return age_days, False, badge, ""
    day_word = "day" if age_days == 1 else "days"
    badge = (
        f'<span class="fresh-badge fresh-stale">&#9888; Data is {age_days} {day_word} old '
        f"&middot; as at {asof_str}</span>"
    )
    banner = (
        '<div class="stale-banner">&#9888; <strong>Stale data warning:</strong> '
        f"this dashboard was built from an export dated <strong>{asof_str}</strong> "
        f"({age_days} {day_word} old), not today. The automated portal fetch likely "
        "failed (expired login session) and the run fell back to the last saved export. "
        "Re-run <strong>Run_BNY_Dashboard.bat</strong> and sign in to the portal when "
        "prompted to refresh the data.</div>"
    )
    return age_days, True, badge, banner


def build_html(m, xlsx_url, pdf_url, history=None):
    asof_str = m["asof"].strftime("%d %b %Y")
    _age_days, _is_stale, fresh_badge, stale_banner = _freshness(m["asof"])
    ctx_json = (
        json.dumps(_ai_context(m, xlsx_url, pdf_url), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    ai_script = AI_ASSISTANT_SCRIPT.replace("__CTX__", ctx_json).replace("__COPILOT_URL__", COPILOT_URL)
    if history is None:
        history = load_history()
    trend = trend_html(m, history)
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
  .page {{ max-width:100%; margin:24px auto; background:var(--surface); box-shadow:0 4px 12px rgba(0,0,0,.08); }}
  .page + .page {{ margin-top:32px; }}
  .header {{ background:var(--navy); color:#fff; padding:16px 28px; }}
  .header h1 {{ font-size:1.35rem; font-weight:700; }}
  .header .sub {{ font-size:.8rem; color:#c7d2e0; margin-top:2px; }}
  .fresh-badge {{ display:inline-block; margin-top:6px; padding:3px 10px; border-radius:9999px; font-size:.72rem; font-weight:700; }}
  .fresh-ok {{ background:rgba(22,163,74,.18); color:#bbf7d0; border:1px solid rgba(22,163,74,.5); }}
  .fresh-stale {{ background:rgba(220,38,38,.2); color:#fecaca; border:1px solid rgba(220,38,38,.6); }}
  .stale-banner {{ margin:0 0 14px; padding:11px 14px; border-radius:8px; background:#fef2f2; border:1px solid #fecaca; border-left:4px solid var(--red); color:#991b1b; font-size:.82rem; line-height:1.45; }}
  .body {{ padding:16px 28px 20px; }}
  .kpi-row {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .kpi {{ border-radius:8px; padding:7px 12px; min-height:48px; display:flex; flex-direction:column; justify-content:space-between; }}
  .kpi-label {{ font-size:.68rem; font-weight:700; opacity:.95; }}
  .kpi-value {{ font-size:1.2rem; font-weight:800; }}
  .grid2 {{ display:grid; grid-template-columns:auto 1fr; gap:24px; margin-bottom:12px; align-items:start; }}
  .risk-matrix-block {{ max-width:420px; margin-bottom:0; }}
  .watchlist-block {{ margin-bottom:12px; }}
  .section-title {{ font-size:.95rem; font-weight:700; margin-bottom:6px; }}
  table {{ border-collapse:collapse; width:100%; font-size:.78rem; }}
  table.matrix td, table.matrix th {{ border:1px solid var(--border); padding:6px 6px; }}
  table.matrix th {{ background:#f5f5f5; font-weight:700; text-align:center; }}
  table.matrix .rowhead {{ font-weight:700; text-align:center; background:#fff; }}
  table.data th, table.data td {{ border:1px solid var(--border); padding:6px 8px; text-align:left; vertical-align:middle; }}
  table.data th {{ background:#f5f5f5; font-weight:700; }}
  table.watchlist {{ table-layout:fixed; font-size:.74rem; }}
  table.watchlist th, table.watchlist td {{ padding:5px 7px; word-wrap:break-word; overflow-wrap:break-word; }}
  table.watchlist th:nth-child(1), table.watchlist td:nth-child(1) {{ width:9%; }}
  table.watchlist th:nth-child(2), table.watchlist td:nth-child(2) {{ width:8%; }}
  table.watchlist th:nth-child(3), table.watchlist td:nth-child(3) {{ width:5%; }}
  table.watchlist th:nth-child(4), table.watchlist td:nth-child(4) {{ width:5%; }}
  table.watchlist th:nth-child(5), table.watchlist td:nth-child(5) {{ width:20%; }}
  table.watchlist th:nth-child(6), table.watchlist td:nth-child(6) {{ width:13%; }}
  table.watchlist th:nth-child(7), table.watchlist td:nth-child(7) {{ width:10%; }}
  table.watchlist th:nth-child(8), table.watchlist td:nth-child(8) {{ width:12%; }}
  table.watchlist th:nth-child(9), table.watchlist td:nth-child(9) {{ width:12%; }}
  table.watchlist th:nth-child(10), table.watchlist td:nth-child(10) {{ width:6%; }}
  .pill {{ display:inline-block; min-width:20px; padding:2px 7px; border-radius:9999px; color:#fff; font-weight:700; font-size:.72rem; }}
  .callout {{ margin:6px 0 8px; font-size:.9rem; font-weight:700; color:#b45309; }}
  .driver-strip {{ margin-top:10px; font-size:.78rem; color:var(--muted); padding:8px 12px; background:#f8fafc; border-left:3px solid var(--amber); border-radius:4px; }}
  .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:20px; }}
  .chart-card {{ border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
  .chart-note {{ font-size:.68rem; color:var(--muted); margin-top:4px; }}
  .score-panel {{ border:1px solid var(--border); border-radius:8px; padding:16px 20px; }}
  .score-h {{ font-size:.9rem; font-weight:700; margin-bottom:10px; }}
  .drivers-row {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
  .driver-item {{ background:#f8fafc; border:1px solid var(--border); border-radius:6px; padding:6px 11px; font-size:.72rem; font-weight:600; color:var(--text); white-space:nowrap; }}
  .driver-item.total {{ background:#fff7ed; border-color:var(--amber); color:#b45309; font-weight:800; }}
  .footer-note {{ font-size:.68rem; color:var(--muted); padding:6px 28px 20px; }}
  .appendix-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:start; }}
  .final-calc {{ margin-top:16px; font-size:1rem; font-weight:700; }}
  .final-score {{ display:inline-block; color:#fff; padding:4px 14px; border-radius:8px; margin-left:6px; }}
  table.open-cases {{ table-layout:fixed; font-size:.72rem; }}
  table.open-cases th, table.open-cases td {{ padding:4px 6px; word-wrap:break-word; overflow-wrap:break-word; }}
  table.open-cases thead {{ display:table-header-group; }}
  table.open-cases tr {{ page-break-inside:avoid; }}
  table.open-cases th:nth-child(1), table.open-cases td:nth-child(1) {{ width:7%; }}
  table.open-cases th:nth-child(2), table.open-cases td:nth-child(2) {{ width:6%; }}
  table.open-cases th:nth-child(3), table.open-cases td:nth-child(3) {{ width:4%; }}
  table.open-cases th:nth-child(4), table.open-cases td:nth-child(4) {{ width:4%; }}
  table.open-cases th:nth-child(5), table.open-cases td:nth-child(5) {{ width:20%; }}
  table.open-cases th:nth-child(6), table.open-cases td:nth-child(6) {{ width:8%; }}
  table.open-cases th:nth-child(7), table.open-cases td:nth-child(7) {{ width:5%; }}
  table.open-cases th:nth-child(8), table.open-cases td:nth-child(8) {{ width:9%; }}
  table.open-cases th:nth-child(9), table.open-cases td:nth-child(9) {{ width:8%; }}
  table.open-cases th:nth-child(10), table.open-cases td:nth-child(10) {{ width:6%; }}
  table.open-cases th:nth-child(11), table.open-cases td:nth-child(11) {{ width:7%; }}
  table.open-cases th:nth-child(12), table.open-cases td:nth-child(12) {{ width:8%; }}
  table.open-cases th:nth-child(13), table.open-cases td:nth-child(13) {{ width:8%; }}
  /* Attention Score trend block (Idea 2b) */
  .trend-block {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap; border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin-bottom:14px; background:#f8fafc; }}
  .trend-score {{ font-size:2rem; font-weight:800; line-height:1; }}
  .trend-meta {{ flex:1; min-width:180px; }}
  .trend-line1 {{ font-size:.82rem; font-weight:700; }}
  .trend-line2 {{ font-size:.8rem; margin-top:3px; }}
  .trend-delta {{ font-weight:800; }}
  .trend-line3 {{ font-size:.68rem; color:var(--muted); margin-top:3px; }}
  .trend-note {{ font-size:.78rem; color:var(--muted); }}
  .trend-spark {{ flex-shrink:0; }}
  /* Appendix B interactive controls (Idea 2a) */
  .ob-toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:10px; }}
  .ob-toolbar input, .ob-toolbar select {{ font:inherit; font-size:.8rem; padding:6px 10px; border:1px solid var(--border); border-radius:6px; background:#fff; color:var(--text); }}
  .ob-toolbar input {{ flex:1; min-width:200px; }}
  .ob-count {{ font-size:.72rem; color:var(--muted); }}
  table.open-cases th.sortable {{ cursor:pointer; user-select:none; }}
  table.open-cases th.sortable:hover {{ background:#ececec; }}
  table.open-cases th[data-sort="asc"]::after {{ content:" \u2191"; font-weight:700; }}
  table.open-cases th[data-sort="desc"]::after {{ content:" \u2193"; font-weight:700; }}
  /* On-screen info pane (hidden when printing / in the PDF) */
  .app-shell {{ display:flex; gap:22px; max-width:1520px; margin:0 auto; padding:24px 18px; align-items:flex-start; }}
  .pages-wrap {{ flex:1; min-width:0; }}
  .pages-wrap .page {{ margin:0 0 32px; }}
  .info-pane {{ width:300px; flex-shrink:0; position:sticky; top:24px; display:flex; flex-direction:column; gap:16px; max-height:calc(100vh - 48px); overflow-y:auto; }}
  .info-pane::-webkit-scrollbar {{ width:5px; }}
  .info-pane::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:9999px; }}
  .info-card {{ background:var(--surface); border-radius:10px; border:1px solid var(--border); box-shadow:0 4px 12px rgba(0,0,0,.08); padding:16px; }}
  .info-card-title {{ font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--border); }}
  .info-pane-head {{ background:var(--navy); color:#fff; border-radius:10px; padding:14px 16px; cursor:pointer; display:flex; align-items:center; gap:10px; user-select:none; }}
  .info-pane-head .head-text {{ flex:1; }}
  .info-pane-head h2 {{ font-size:.95rem; font-weight:700; }}
  .info-pane-head p {{ font-size:.72rem; color:#c7d2e0; margin-top:3px; }}
  .info-pane-toggle {{ flex-shrink:0; font-size:.9rem; line-height:1; transition:transform .2s ease; }}
  .info-pane.collapsed {{ width:auto; }}
  .info-pane.collapsed .info-card {{ display:none; }}
  .info-pane.collapsed .head-text {{ display:none; }}
  .info-pane.collapsed .info-pane-head {{ padding:10px 12px; border-radius:9999px; }}
  .info-pane.collapsed .info-pane-toggle {{ transform:rotate(-90deg); }}
  .guide-step {{ display:flex; gap:10px; margin-bottom:11px; }}
  .guide-step:last-child {{ margin-bottom:0; }}
  .guide-num {{ flex-shrink:0; width:22px; height:22px; border-radius:50%; background:var(--navy); color:#fff; font-size:.72rem; font-weight:700; display:flex; align-items:center; justify-content:center; }}
  .guide-text {{ font-size:.78rem; line-height:1.42; }}
  .tech-block {{ margin-bottom:14px; }}
  .tech-block:last-child {{ margin-bottom:0; }}
  .tech-block-title {{ font-size:.8rem; font-weight:700; margin-bottom:6px; color:var(--navy); }}
  .tech-item {{ font-size:.75rem; line-height:1.45; color:var(--muted); margin-bottom:5px; }}
  .info-code {{ font-family:Consolas,'Courier New',monospace; background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:.72rem; color:var(--navy); }}
  .dash-link {{ display:inline-block; margin-top:8px; padding:7px 12px; background:var(--navy); color:#fff; font-size:.76rem; font-weight:600; text-decoration:none; border-radius:6px; }}
  .dash-link:hover {{ background:#0e1f39; }}
  /* Responsive screen layout (Idea 1) - A4 sizing stays under @media print */
  @media screen and (max-width:1024px) {{
    .app-shell {{ flex-direction:column; max-width:100%; padding:14px; }}
    .info-pane {{ width:100%; position:static; max-height:none; flex-direction:row; flex-wrap:wrap; }}
    .info-pane .info-card {{ flex:1 1 240px; }}
    .info-pane-head {{ width:100%; }}
    .pages-wrap {{ width:100%; }}
    .page {{ max-width:100%; }}
  }}
  /* "Ask AI" assistant (screen only) */
  .ai-fab {{ position:fixed; right:22px; top:22px; z-index:50; background:var(--navy); color:#fff; border:none; border-radius:9999px; padding:12px 18px; font:inherit; font-size:.85rem; font-weight:700; box-shadow:0 6px 18px rgba(0,0,0,.25); cursor:pointer; display:flex; align-items:center; gap:8px; }}
  .ai-fab:hover {{ background:#0e1f39; }}
  .ai-panel {{ position:fixed; right:22px; top:76px; z-index:50; width:360px; max-width:calc(100vw - 44px); max-height:calc(100vh - 110px); background:var(--surface); border:1px solid var(--border); border-radius:14px; box-shadow:0 12px 36px rgba(0,0,0,.28); display:none; flex-direction:column; overflow:hidden; }}
  .ai-panel.open {{ display:flex; }}
  .ai-head {{ background:var(--navy); color:#fff; padding:13px 16px; display:flex; align-items:center; gap:10px; }}
  .ai-head h3 {{ font-size:.95rem; font-weight:700; flex:1; }}
  .ai-head .ai-close {{ background:none; border:none; color:#c7d2e0; font-size:1.3rem; cursor:pointer; line-height:1; }}
  .ai-body {{ padding:14px 16px; overflow-y:auto; }}
  .ai-intro {{ font-size:.76rem; color:var(--muted); line-height:1.45; margin-bottom:13px; }}
  .ai-presets-title, .ai-ask-title {{ font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin:0 0 7px; }}
  .ai-preset {{ display:block; width:100%; text-align:left; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:8px 11px; font:inherit; font-size:.78rem; color:var(--text); cursor:pointer; margin-bottom:7px; }}
  .ai-preset:hover {{ background:#eef2f7; border-color:#cbd5e1; }}
  .ai-ask {{ margin-top:14px; }}
  .ai-ask textarea {{ width:100%; min-height:62px; resize:vertical; font:inherit; font-size:.8rem; padding:9px 11px; border:1px solid var(--border); border-radius:8px; }}
  .ai-actions {{ display:flex; gap:8px; margin-top:9px; }}
  .ai-btn {{ flex:1; border:none; border-radius:8px; padding:9px 12px; font:inherit; font-size:.78rem; font-weight:700; cursor:pointer; }}
  .ai-btn-primary {{ background:var(--navy); color:#fff; }}
  .ai-btn-primary:hover {{ background:#0e1f39; }}
  .ai-btn-ghost {{ background:#f1f5f9; color:var(--text); border:1px solid var(--border); }}
  .ai-note {{ font-size:.68rem; color:var(--muted); margin-top:10px; line-height:1.4; }}
  .ai-answer {{ margin:12px 0 4px; padding:13px 14px; border-radius:10px; background:#f8fafc; border:1px solid var(--border); border-left:4px solid var(--navy); font-size:.8rem; color:var(--text); line-height:1.5; }}
  .ai-answer .ai-answer-q {{ font-weight:700; color:var(--navy); margin-bottom:8px; }}
  .ai-answer ul, .ai-answer ol {{ margin:6px 0 6px 18px; padding:0; }}
  .ai-answer li {{ margin-bottom:5px; }}
  .ai-answer p {{ margin:0 0 7px; }}
  .ai-answer .ai-sub {{ color:var(--muted); font-size:.74rem; }}
  .ai-answer .ai-answer-src {{ margin-top:9px; padding-top:8px; border-top:1px dashed var(--border); font-size:.66rem; color:var(--muted); }}
  .ai-answer .ai-draft {{ width:100%; min-height:150px; resize:vertical; font-family:Consolas,'Courier New',monospace; font-size:.72rem; padding:9px 11px; border:1px solid var(--border); border-radius:8px; margin-top:4px; }}
  .ai-status {{ display:none; margin-top:11px; padding:10px 12px; border-radius:8px; background:#ecfdf5; border:1px solid #a7f3d0; border-left:4px solid var(--green); color:#065f46; font-size:.78rem; line-height:1.45; }}
  .ai-status strong {{ color:#064e3b; }}
  .ai-fallback textarea {{ width:100%; min-height:96px; resize:vertical; font-family:Consolas,'Courier New',monospace; font-size:.68rem; padding:9px 11px; border:1px solid var(--border); border-radius:8px; margin-bottom:8px; }}
  .ai-open-link {{ display:block; text-align:center; text-decoration:none; }}
  .ai-toast {{ position:fixed; right:22px; top:76px; z-index:60; background:var(--green); color:#fff; padding:10px 14px; border-radius:8px; font-size:.78rem; font-weight:600; box-shadow:0 6px 18px rgba(0,0,0,.25); opacity:0; transform:translateY(-8px); transition:opacity .2s, transform .2s; pointer-events:none; }}
  .ai-toast.show {{ opacity:1; transform:translateY(0); }}
  @media print {{
    body {{ background:#fff; }}
    .app-shell {{ display:block; padding:0; max-width:none; }}
    .info-pane {{ display:none; }}
    .pages-wrap .page {{ margin:0; }}
    .page {{ box-shadow:none; margin:0; max-width:none; page-break-after:always; }}
    .page:last-child {{ page-break-after:auto; }}
    .ob-toolbar {{ display:none !important; }}
    table.open-cases th.sortable {{ cursor:default; }}
    .ai-fab, .ai-panel, .ai-toast {{ display:none !important; }}
    @page {{ size:A4 landscape; margin:8mm; }}
  }}</style>
</head>
<body>
<div class="app-shell">

  <!-- LEFT INFO PANE (screen only) -->
  <aside class="info-pane" id="infoPane">
    <div class="info-pane-head" onclick="document.getElementById('infoPane').classList.toggle('collapsed')">
      <span class="info-pane-toggle">&#9660;</span>
      <div class="head-text">
        <h2>Dashboard Guide</h2>
        <p>How to run &amp; how it works</p>
      </div>
    </div>

    <div class="info-card">
      <div class="info-card-title">Dashboard Access</div>
      <div class="tech-item">This dashboard can be accessed via the external BNY SharePoint site.</div>
      <a class="dash-link" href="https://vfmcorp.sharepoint.com/sites/External-VFMCBNYMellon/Shared%20Documents/General/Operations/VFMC%20Daily%20Scorecard/BNY_Executive_Dashboard_Latest.html" target="_blank" rel="noopener">&#128202; Open the Dashboard</a>
    </div>

    <div class="info-card">
      <div class="info-card-title">User Guide &ndash; Running the Dashboard to retrieve latest BNY Portal Ticket Data</div>
      <div class="guide-step"><div class="guide-num">1</div><div class="guide-text">You must be logged into the <strong>Citizen Dev Box</strong> to fetch the data.</div></div>
      <div class="guide-step"><div class="guide-num">2</div><div class="guide-text">Ensure the SharePoint and OneDrive have been synchronised (the <strong>3 dots</strong> selection, top-right corner of the SharePoint folder) &ndash; <a href="https://vfmcorp.sharepoint.com/:f:/r/sites/External-VFMCBNYMellon/Shared%20Documents/General/Operations/VFMC%20Daily%20Scorecard?d=wbdcc928896e14568a0a663a001bf4309&amp;csf=1&amp;web=1&amp;e=sxHdsF" target="_blank" rel="noopener">VFMC Daily Scorecard</a>.</div></div>
      <div class="guide-step"><div class="guide-num">3</div><div class="guide-text">Open the <strong>VFMC Daily Scorecard</strong> folder in File Explorer (synced from SharePoint/Teams).</div></div>
      <div class="guide-step"><div class="guide-num">4</div><div class="guide-text">Double-click <span class="info-code">Run_BNY_Dashboard.bat</span>.</div></div>
      <div class="guide-step"><div class="guide-num">5</div><div class="guide-text">It fetches the latest cases from the BNY (Eagle) portal. If your saved session has expired, it asks for your portal email (press <strong>Enter</strong> to accept the default) then your <strong>password</strong>.</div></div>
      <div class="guide-step"><div class="guide-num">6</div><div class="guide-text">The dashboard runs: <em>fetch &rarr; HTML &rarr; PDF &rarr; archive</em>.</div></div>
      <div class="guide-step"><div class="guide-num">7</div><div class="guide-text">This page opens automatically as the latest dashboard.</div></div>
      <div class="guide-step"><div class="guide-num">8</div><div class="guide-text">The dated PDF and source data are moved to the <span class="info-code">Archive</span> subfolder &ndash; download links are in <strong>Appendix B</strong>.</div></div>
    </div>

    <div class="info-card">
      <div class="info-card-title">Data &amp; Technical Information</div>

      <div class="tech-block">
        <div class="tech-block-title">&#9881;&#65039; How It Works</div>
        <div class="tech-item">A self-service <strong>citizen-developer</strong> solution &ndash; a set of Python scripts, no dedicated server or licence.</div>
        <div class="tech-item">Each run rebuilds this single-file HTML report plus a print-ready PDF from live case data.</div>
      </div>

      <div class="tech-block">
        <div class="tech-block-title">&#128229; Data Sourcing</div>
        <div class="tech-item"><span class="info-code">fetch_latest_export.py</span> uses Playwright to sign in to the Eagle client portal and export the open-case list to Excel.</div>
        <div class="tech-item">Your password is entered at runtime and <strong>never stored</strong>; the login session is cached until it expires.</div>
      </div>

      <div class="tech-block">
        <div class="tech-block-title">&#129518; Scoring</div>
        <div class="tech-item">The Management Attention Score weights open priorities, case ageing and repeat-incident themes (see <strong>Appendix A</strong>).</div>
      </div>

      <div class="tech-block">
        <div class="tech-block-title">&#128451;&#65039; Outputs &amp; Archive</div>
        <div class="tech-item"><span class="info-code">BNY_Executive_Dashboard_Latest.html</span> is the always-current view in the main folder.</div>
        <div class="tech-item">Dated PDF and XLSX are archived and links surfaced in Appendix B; the folder syncs via SharePoint/Teams.</div>
      </div>

      <div class="tech-block">
        <div class="tech-block-title">&#128295; Built With</div>
        <div class="tech-item">Python (pandas, Playwright, openpyxl). Source on <a href="https://github.com/TSmith-VFMC/Customised-Survey-Dashboard" target="_blank" rel="noopener">GitHub</a>.</div>
      </div>
    </div>
  </aside>

  <div class="pages-wrap">

  <!-- PAGE 1 -->
  <div class="page">
    <div class="header">
      <h1>BNY Managed Data Services &ndash; Executive Service Health</h1>
      <div class="sub">Point-in-time daily view | As at {asof_str}</div>
      <div>{fresh_badge}</div>
    </div>
    <div class="body">
      {stale_banner}
      <div class="grid2">
        <div class="risk-matrix-block">
          <div class="section-title">Priority &times; Age Risk Matrix</div>
          {risk_matrix_html(m)}
        </div>
        <div>
          {kpi_tiles_html(m)}
        </div>
      </div>
      <div class="watchlist-block">
        <div class="section-title">Executive Watchlist &ndash; Top 5</div>
        {watchlist_html(m)}
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
      <div class="section-title">Attention Score Trend</div>
      {trend}
      <div class="section-title">Management Attention Score</div>
      {scoring_html(m)}
      <div class="section-title" style="margin-top:22px">Pending RCA &ndash; Root Cause Analysis in Progress ({m['rca_pending']})</div>
      <div class="chart-note" style="margin-bottom:8px">These open cases are parked awaiting root-cause analysis and are excluded from the Management Attention Score, KPIs, risk matrix and Executive Watchlist.</div>
      {rca_table_html(m)}
    </div>
  </div>

  <!-- PAGE 3 - APPENDIX A -->
  <div class="page">
    <div class="header">
      <h1>Appendix A &ndash; Management Attention Score Methodology</h1>
      <div class="sub">Full transparency of how today's score ({m['score']} &middot; {m['band']}) is calculated | As at {asof_str}</div>
    </div>
    <div class="body">
      {appendix_html(m)}
    </div>
  </div>

  <!-- PAGE 4 - APPENDIX B -->
  <div class="page">
    <div class="header">
      <h1>Appendix B &ndash; Full Open Case Listing</h1>
      <div class="sub">As at {asof_str}</div>
    </div>
    <div class="body">
      {appendix_b_html(m, xlsx_url, pdf_url)}
    </div>
    <div class="footer-note">
      Source & generator: <a href="https://github.com/TSmith-VFMC/Customised-Survey-Dashboard" target="_blank" rel="noopener">github.com/TSmith-VFMC/Customised-Survey-Dashboard</a>
    </div>
  </div>

  </div><!-- /pages-wrap -->
</div><!-- /app-shell -->

<!-- ASK AI ASSISTANT (screen only) -->
<button class="ai-fab" id="aiFab" onclick="aiToggle()" aria-label="Open dashboard assistant">
  <span aria-hidden="true">&#129302;</span> Ask AI
</button>
<div class="ai-panel" id="aiPanel" role="dialog" aria-label="Dashboard assistant">
  <div class="ai-head">
    <span aria-hidden="true">&#129302;</span>
    <h3>Dashboard Assistant</h3>
    <button class="ai-close" onclick="aiToggle()" aria-label="Close assistant">&times;</button>
  </div>
  <div class="ai-body">
    <div class="ai-intro">
      Ask about today's data (as at {asof_str}). Answers are generated <strong>directly from this
      dashboard's figures</strong> &ndash; nothing is sent anywhere and no credentials are used. For
      open-ended analysis you can also hand the question and data to Microsoft&nbsp;Copilot.
    </div>
    <div class="ai-presets-title">Standard questions &ndash; click for an instant answer</div>
    <div id="aiPresets"></div>
    <div class="ai-answer" id="aiAnswer" style="display:none"></div>
    <div class="ai-ask">
      <div class="ai-ask-title">Or ask your own</div>
      <textarea id="aiQuestion" placeholder="e.g. How many P1 cases are open? Which case is the biggest risk?"></textarea>
      <div class="ai-actions">
        <button class="ai-btn ai-btn-primary" onclick="aiAnswer()">Answer</button>
        <button class="ai-btn ai-btn-ghost" onclick="aiAsk()">Ask Copilot</button>
        <button class="ai-btn ai-btn-ghost" onclick="aiCopy()">Copy prompt</button>
      </div>
      <div class="ai-note">&ldquo;Answer&rdquo; replies instantly from today's data. &ldquo;Ask Copilot&rdquo; opens Copilot with the question + data on your clipboard &ndash; paste with Ctrl+V.</div>
      <div class="ai-status" id="aiStatus"></div>
      <div class="ai-fallback" id="aiFallback" style="display:none">
        <div class="ai-ask-title" style="margin-top:12px">Manual copy (auto-copy was blocked)</div>
        <textarea id="aiPromptOut" readonly aria-label="Generated prompt to copy"></textarea>
        <a class="ai-btn ai-btn-primary ai-open-link" id="aiOpenLink" href="#" target="_blank" rel="noopener">Open Copilot</a>
      </div>
    </div>
  </div>
</div>
<div class="ai-toast" id="aiToast">Prompt copied</div>
{ai_script}
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

    history = record_history(m, asof)

    archive_url = SHAREPOINT_FOLDER_URL + "Archive/"
    xlsx_url = archive_url + quote(os.path.basename(source_path))
    pdf_name = f'BNY_Executive_Dashboard_Services_{asof.strftime("%Y-%m-%d")}.pdf'
    pdf_url = archive_url + quote(pdf_name)
    html = build_html(m, xlsx_url, pdf_url, history=history)
    out_name = "BNY_Executive_Dashboard_Latest.html"
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
