# BNY Dashboard — Enhancement Backlog

Tracked ideas for future enhancements to the dashboard app. Not yet started.

---

## Idea 1 — Embed into a SharePoint page or Teams tab

**Goal:** Let people *view the dashboard in the browser* instead of opening the HTML file.

**Two parts:**

- **Config (no code):** add an **Embed** / **File viewer** web part on a SharePoint page,
  or a **Website / Teams tab** pointing at the Latest HTML URL
  (`.../VFMC Daily Scorecard/BNY_Executive_Dashboard_Latest.html`).
- **Code to make it embed cleanly:** on screen the report is a fixed A4 width (1180px),
  which looks cramped in an iframe. Add a **responsive screen layout** — fluid width on
  screen, with the A4 sizing applied only under `@media print`, so it fills an embedded
  frame nicely (also improves laptop/phone viewing).

**Effort:** Low–medium.
**Value:** Easier access; better appearance on laptops/phones and inside embeds.
**Notes:** Must not change the print/PDF output (A4 sizing stays under `@media print`).

---

## Idea 2 — High-value dashboard features

### 2a. Make Appendix B interactive on screen

Add a **search box**, **click-to-sort column headers**, and a **priority/RAG filter** to the
full open-case table. Execs could instantly find "all P2s" or search a case number.

- Pure client-side JS, no backend.
- Screen-only — zero impact on the PDF (hidden via `@media print`).

**Effort:** Medium.
**Value:** Big usability win for exploring the open-case list.

### 2b. Management Attention Score trend over time

Right now every view is point-in-time. If each run appends
`{date, score, backlog, P1, agedP2}` to a small `history.json` in the folder, we can draw a
mini line/sparkline, e.g. *"Score: 85 ▲ from 78 last run."*

- Turns a snapshot into a trend — the single most exec-valuable add.
- Requires the pipeline to write/append a tiny history file each run.

**Effort:** Medium.
**Value:** High — adds trend/context that a point-in-time snapshot can't.

---

## Idea 3 — Automated email via Power Automate (Phase 1)

**Goal:** Automatically email the latest dashboard PDF after each run, using a Power
Automate **cloud flow** (no admin, no local Outlook, no stored credentials, no Task
Scheduler).

A cloud flow watches the SharePoint `Archive` folder with a **"When a file is created"**
trigger, filters to the dashboard PDF, gets the file content, and sends it via
**Send an email (V2)**.

**Full step-by-step setup guide:** see
[power-automate-email-flow.md](power-automate-email-flow.md).

**Effort:** Low (built in the Power Automate web UI, ~6 steps).
**Value:** Removes the manual emailing step.
**Notes:** Phases 2–3 (automating the run itself on the Citizen Dev Box via Power Automate
Desktop) are parked pending specialist sign-off on unattended portal login — covered in the
same doc's roadmap.

