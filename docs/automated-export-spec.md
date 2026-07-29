# Spec: Automated Ticket Export from the Eagle Client Portal

**Status:** Proposed
**Owner:** T. Smith
**Related components:** `generate_bny_dashboard.py`, `generate_bny_dashboard_html.py`, `Run_BNY_Dashboard.bat`

---

## 1. Problem / Background

The dashboards are generated from a ServiceNow-style Excel export
(`VFM_Cases_<timestamp>.xlsx`) that is currently **downloaded by hand** from the
Eagle Investment Systems client portal. Each refresh requires a person to log in,
click the export icon, and drop the file into the source folder before running the
generators.

The portal's export icon is backed by a single REST endpoint that returns an Excel
file directly:

```
GET https://xpclientportal.eagleinvsys.com/api/v1/tickets
    ?export=EXCEL
    &sortBy=updatedAt
    &isDesc=1
    &selectedColumns=number,state,environment,sid,priority,shortDescription,
                     openedAt,updatedAt,openedBy,customerNumber,caseType,
                     bugTrackerNumber,subcategory,customerKeyword,category,issueType
    &filterValues={}
```

These `selectedColumns` map 1:1 to the fields `load_cases()` already consumes
(Number, State, Priority, Opened, Updated, Customer Keyword, etc.), so if the
endpoint can be called programmatically, the manual download step is eliminated.

The blocker is **authentication**: the endpoint is gated by the session the browser
obtains after an interactive login (SSO + MFA).

## 2. Goal

Automatically obtain the latest ticket export and place it in the generators' source
folder, with **no credentials stored in code or the repository**, so a single command
(or scheduled task) can refresh the dashboards end to end.

### Non-goals
- Storing or transmitting the user's password / MFA secrets through any script.
- Fully unattended (headless, zero-touch) automation on day one — see Open Questions.
- Changes to the scoring model, chart logic, or output formats.

## 3. Proposed Solution — Playwright persistent session

Use Playwright (Python) with a **persistent browser profile** so the interactive
SSO/MFA login is performed once and the resulting session is reused.

### Flow
1. Launch Chromium with a persistent `user_data_dir` stored locally (git-ignored).
2. **First run / expired session:** open the portal in a headed window; the user
   completes login + MFA manually. Cookies persist in the profile.
3. Call the export endpoint **using the authenticated session**
   (`context.request.get(EXPORT_URL)`), which streams back the `.xlsx` bytes.
4. Save the file into `CONFIG["source_dir"]` using the existing filename pattern
   `VFM_Cases_MM-DD-YYYY_H_MM_AMPM.xlsx` (current local time), so the generators'
   "newest file" selection picks it up automatically.
5. **Subsequent runs:** the stored session is reused silently. On a `401`/redirect to
   login, the window re-opens for a quick re-auth.

### Why this approach
- Handles **SSO + MFA** cleanly (human does it once, interactively).
- **No secrets in code** — only a local browser profile, which is git-ignored.
- Reuses the existing file-based pipeline; the generators are unchanged.

### Alternatives considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Token/cookie capture + `requests`** | Simplest code | Token expires; manual re-capture from DevTools each time |
| **Service-account / API key** (ask Eagle/BNY) | Cleanest for scheduled, unattended runs | Depends on the vendor offering one; procurement/approval |
| **Playwright persistent session** *(chosen)* | Handles MFA, no stored secrets, reuses session | First run and periodic re-login are interactive |

## 4. Implementation Sketch

New module: `fetch_latest_export.py`

- `EXPORT_URL` constant (the endpoint above).
- `PORTAL_HOME` constant for the login landing page.
- `AUTH_DIR = ".auth/"` for the persistent profile (git-ignored).
- `fetch_export(headed=True) -> Path`:
  1. `p = sync_playwright().start()`
  2. `ctx = p.chromium.launch_persistent_context(AUTH_DIR, headless=False)`
  3. Navigate to `PORTAL_HOME`; if redirected to login, wait for the user to finish.
  4. `resp = ctx.request.get(EXPORT_URL)`; assert `resp.ok` and content-type is a
     spreadsheet (guard against being handed an HTML login page).
  5. Write bytes to `source_dir / f"VFM_Cases_{now:%m-%d-%Y_%-I_%M_%p}.xlsx"`.
  6. Return the path; close context.
- Optional `--headless` retry: attempt silent download first; fall back to headed
  login only when the session is invalid.
- Wire into `Run_BNY_Dashboard.bat` **before** the two generator calls.

### Dependencies
- Add `playwright>=1.44` to `requirements.txt`.
- One-time: `playwright install chromium`.

### Security & governance
- `.auth/` (browser profile / cookies) and any `*.xlsx` stay **git-ignored**.
- No username, password, or MFA material is ever read, logged, or stored by the script.
- Prefer a **read-only** portal account if BNY can provide one.
- Confirm automated export is permitted under BNY/Eagle acceptable-use policy.
- Validate the downloaded payload is a real spreadsheet before overwriting anything.

## 5. Acceptance Criteria

- [ ] Running `fetch_latest_export.py` produces a valid `VFM_Cases_*.xlsx` in the
      source folder that opens in the generators without changes.
- [ ] First run prompts an interactive login; a second run within the session window
      downloads **without** prompting.
- [ ] No credentials or tokens appear in the repo, logs, or command output.
- [ ] `Run_BNY_Dashboard.bat` performs fetch → HTML → PPTX in one invocation.
- [ ] A clear, actionable error is shown when the session has expired.

## 6. Open Questions
1. Does login present an **MFA** challenge, and how often is re-auth forced
   (every visit / daily / weekly)? Drives how "hands-off" scheduling can be.
2. Does Eagle/BNY offer an **API token or service account** for this portal? If so,
   the service-account option supersedes the browser approach for scheduled runs.
3. Are there **filters** we should bake into `filterValues` (e.g. environment,
   customer) rather than exporting everything?
4. Preferred **cadence** — on demand, or a scheduled Windows Task?

## 7. Rollout Plan
1. Build `fetch_latest_export.py` behind its own entry point (no changes to existing
   generators).
2. Manual validation over a few refresh cycles to confirm session longevity.
3. Add the batch-file wiring once stable.
4. Revisit the service-account/API-token path for fully scheduled automation.
