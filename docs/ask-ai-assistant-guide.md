# BNY Dashboard — "Ask AI" Assistant: How-To Guide

**What it is:** a built-in AI helper inside the BNY Managed Data Services dashboard.
Click one button, ask a question about today's figures (or pick a ready-made prompt), and
it hands your question — together with the dashboard's data — straight to Microsoft Copilot
for an instant, plain-English answer. No new software, no passwords, no data leaving the
approved Microsoft 365 environment.

---

## 1. Where to find it

Open the dashboard (`BNY_Executive_Dashboard_Latest.html`) in your browser. In the
**top-right corner** you'll see a navy pill button:

> 🤖 **Ask AI**

It stays pinned to that corner as you scroll. It was deliberately placed top-right so it
isn't confused with the browser's own Copilot icon (which usually sits bottom-right).

**Note:** the button only appears in the live HTML dashboard. It is intentionally hidden in
the printed/emailed PDF, so the PDF stays clean for distribution.

---

## 2. How to use it (30 seconds)

1. Click **🤖 Ask AI** in the top-right corner. A small **Dashboard Assistant** panel opens.
2. Either:
   - **Pick a standard prompt** — click one of the ready-made questions, or
   - **Type your own** question in the box (e.g. *"Which case is the biggest risk and why?"*).
3. Click **Ask Copilot**. The assistant copies a complete prompt to your clipboard and opens
   Microsoft Copilot in a new browser tab.
4. In the Copilot tab, click into the message box and **paste** (press **Ctrl + V**), then
   press Enter. Copilot answers using the dashboard's data.
   - *(Prefer to review first? Click **Copy prompt** instead — it copies the prompt without
     opening Copilot, so you can paste it wherever you like.)*

That's it. You can repeat with as many questions as you like.

---

## 3. The standard prompts

To save typing, the panel includes one-click prompts for the most common executive questions:

- **Summarise today's service health** in 3 short bullets for an executive.
- **Explain why the Management Attention Score is _NN_ today**, using the score drivers.
- **Which open cases are the biggest risks** right now, and why? (top 3)
- **Draft a short status email to the client** about today's position.
- **What are the repeat incident themes** and what root-cause actions do they suggest?

Clicking one drops it into the question box — you can tweak it before sending if you want.

---

## 4. What the assistant "knows" (the knowledge layer)

When you ask a question, the dashboard automatically bundles a snapshot of **today's data**
into the prompt, so Copilot answers from real figures rather than guessing. That snapshot
includes:

- The headline numbers — **Management Attention Score & band**, backlog, P1 count, aged P2s,
  repeat incidents, RCA-pending count, and open cases by priority.
- The **score drivers** and the **MTTR (resolution-time) trend**.
- The **Top-5 Executive Watchlist** and the **repeat incident themes**.
- The **full list of open cases** — case number, priority, days open, subject, state and
  assignee.
- **Links** to the full Excel source data and the PDF report, so you can ask Copilot to point
  you to them for deeper detail.

Because the data travels with the question, answers are specific to the day you're viewing.

---

## 5. Why it's safe

This was designed to fit a corporate, data-sensitive environment:

- **No stored credentials and no API keys.** Nothing secret is embedded in the dashboard file,
  so it's safe to share on SharePoint.
- **Uses Microsoft 365 Copilot** — the organisation's already-approved tool. Your question and
  data go to Copilot under your own M365 account, **staying inside the Microsoft tenant**.
- **You stay in control** — the assistant only prepares the prompt; you choose when to send it,
  and you can review it first with **Copy prompt**.

---

## 6. Good to know / limitations

- **Paste step:** enterprise Copilot can't be pre-filled from a link, so you paste the prompt
  with **Ctrl + V**. The panel reminds you of this.
- **Screen only:** the button never appears in the PDF or when printing.
- **Point-in-time:** the assistant answers from the data in the dashboard you have open. Re-run
  the dashboard to refresh to the latest figures (the top-left data badge shows the date).
- **It's a prototype / "citizen developer" feature** — a lightweight bridge to Copilot, not a
  bespoke chatbot. If richer, fully in-page answers are wanted later, the next step would be an
  in-tenant **Azure OpenAI** connection (a governance decision, as data would be sent to an LLM
  service).

---

## 7. For the technical owner — configuration

The feature lives entirely in **`generate_bny_dashboard_html.py`** (the script that builds the
dashboard). Two things are easy to change:

- **Which Copilot opens** — edit the `COPILOT_URL` constant near the top of the file. It
  currently points to Microsoft 365 Copilot (`https://m365.cloud.microsoft/chat`). Change it if
  your organisation uses a different Copilot entry point.
- **The standard prompts** — edit the `PRESETS` list inside the `AI_ASSISTANT_SCRIPT` section to
  add, remove or reword the one-click questions.

After any change, re-run the dashboard to regenerate the HTML, then run `publish_app.py` to copy
the updated script into the shared `_app` folder.

---

*Built as part of the BNY Daily Scorecard automation. Source and history:
github.com/TSmith-VFMC/Customised-Survey-Dashboard.*
