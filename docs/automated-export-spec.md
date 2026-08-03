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
obtains after an interactive login. The portal uses a **plain username + password**
login (confirmed **no MFA**), which means the login form can be completed
programmatically from a credential store, enabling fully unattended runs.

## 2. Goal

Automatically obtain the latest ticket export and place it in the generators' source
folder, with **no credentials stored in code or the repository**, so a single command
(or scheduled task) can refresh the dashboards end to end.

### Non-goals
- Storing the user's password in code, the repository, logs, or command output.
- Changes to the scoring model, chart logic, or output formats.

## 3. Proposed Solution — Playwright with credential-store login

Because the portal requires only a username + password (no MFA), the login can be
automated end to end. Credentials are read at runtime from **Windows Credential
Manager** via the `keyring` library — they are entered once by the user directly into
their own terminal and never pass through code, the repo, or this assistant.

### Flow
1. Read `username` / `password` from Windows Credential Manager (`keyring`).
2. Launch Chromium (headless by default) with a persistent `user_data_dir` stored
   locally (git-ignored) so a valid session is reused between runs.
3. If not already authenticated, navigate to the portal login page, fill the
   username + password fields, and submit.
4. Call the export endpoint **using the authenticated session**
   (`context.request.get(EXPORT_URL)`), which streams back the `.xlsx` bytes.
5. Save the file into `CONFIG["source_dir"]` using the existing filename pattern
   `VFM_Cases_MM-DD-YYYY_H_MM_AMPM.xlsx` (current local time), so the generators'
   "newest file" selection picks it up automatically.
6. **Subsequent runs:** reuse the stored session; only re-login when it has expired.

### Why this approach
- **No MFA** means it can run **headless and scheduled** with no human in the loop.
- **No secrets in code** — the password lives only in Windows Credential Manager.
- Reuses the existing file-based pipeline; the generators are unchanged.

### Credential setup (one-time, done by the user)
A small helper (`--set-credentials`) prompts for the username and password using
`getpass` and stores them with `keyring.set_password(...)`. The password is typed
directly into the user's terminal; it is never handed to any script argument or to
this assistant.

### Alternatives considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Token/cookie capture + `requests`** | Simplest code | Token expires; manual re-capture from DevTools each time |
| **Service-account / API key** (ask Eagle/BNY) | Cleanest, no browser at all | Depends on the vendor offering one; procurement/approval |
| **Playwright + credential-store login** *(chosen)* | Headless, schedulable, no MFA to satisfy, no secrets in code | Password stored in OS credential vault (acceptable); UI selectors can change |

## 4. Implementation Sketch

New module: `fetch_latest_export.py`

- `EXPORT_URL` constant (the endpoint above).
- `PORTAL_LOGIN` constant for the login page; `LOGIN_SELECTORS` for the username /
  password / submit fields (to be confirmed against the live page).
- `AUTH_DIR = ".auth/"` for the persistent profile (git-ignored).
- `KEYRING_SERVICE = "eagle-xpclientportal"` for credential lookup.
- `set_credentials()` (`--set-credentials`): prompt via `getpass`, store with `keyring`.
- `fetch_export() -> Path`:
  1. Load `username` / `password` from `keyring`.
  2. `ctx = p.chromium.launch_persistent_context(AUTH_DIR, headless=True)`.
  3. Probe the export URL; if it returns a login page/redirect, perform the form login
     and retry.
  4. `resp = ctx.request.get(EXPORT_URL)`; assert `resp.ok` and the content-type is a
     spreadsheet (guard against being handed an HTML login page).
  5. Write bytes to `source_dir / f"VFM_Cases_{now:%m-%d-%Y_%-I_%M_%p}.xlsx"`.
  6. Return the path; close context.
- Wire into `Run_BNY_Dashboard.bat` **before** the two generator calls.

### Dependencies
- Add `playwright>=1.44` and `keyring>=24` to `requirements.txt`.
- One-time: `playwright install chromium`.

### Security & governance
- `.auth/` (browser profile / cookies) and any `*.xlsx` stay **git-ignored**.
- The password is stored **only** in Windows Credential Manager (via `keyring`); it is
  never read into source, logged, printed, passed as a CLI argument, or shared with
  this assistant. It is entered once directly by the user via `getpass`.
- Prefer a **read-only** portal account if BNY can provide one.
- Confirm automated export is permitted under BNY/Eagle acceptable-use policy.
- Validate the downloaded payload is a real spreadsheet before overwriting anything.

## 5. Acceptance Criteria

- [ ] Running `fetch_latest_export.py` produces a valid `VFM_Cases_*.xlsx` in the
      source folder that opens in the generators without changes.
- [ ] With valid stored credentials, the download runs **headless with no prompts**.
- [ ] No credentials or tokens appear in the repo, logs, or command output.
- [ ] `Run_BNY_Dashboard.bat` performs fetch → HTML → PPTX in one invocation.
- [ ] A clear, actionable error is shown when login fails or the session has expired.

## 6. Open Questions
1. ~~Does login present an **MFA** challenge?~~ **Resolved:** no MFA — username +
   password only.
2. Does Eagle/BNY offer an **API token or service account** for this portal? If so,
   it would remove the browser dependency entirely for scheduled runs.
3. ~~Are there **filters** we should bake into `filterValues`~~ **Resolved:** keep
   exporting everything (`filterValues={}`), matching current manual behavior.
4. ~~Preferred **cadence**~~ **Resolved:** on-demand now via `Run_BNY_Dashboard.bat`;
   a scheduled Windows Task is planned as a follow-up (step 4 of rollout, below).
5. ~~Exact **login-page field selectors**~~ **Resolved:** the portal login is a
   PingFederate SSO page (`myeagleapps.eagleinvsys.com/idp/startSSO.ping`) with a
   plain form — `#username` (the user's VFMC email, e.g. `tsmith@vfmc.vic.gov.au`),
   `#password`, and a `a.ping-button.allow` "Sign On" link that submits the form.

## 7. Rollout Plan
1. ~~Build `fetch_latest_export.py`~~ **Done** — `--set-credentials` (keyring +
   getpass) and headless `fetch_export()` implemented; wired into
   `Run_BNY_Dashboard.bat` before the HTML/PPTX generator calls (non-fatal on
   failure — falls back to whatever export is already in the source folder).
2. ~~Capture the login-page selectors~~ **Done** (see Open Question 5).
3. ~~Manual validation over a few refresh cycles~~ **Done** — validated
   end-to-end (headed and headless), producing a valid 15-column, 2500+ row
   `.xlsx` that the generators picked up correctly. Implementation notes from
   getting this working, for future maintenance:
   - Login is a multi-hop SAML redirect; must `page.wait_for_url("**xpclientportal.eagleinvsys.com**")`
     after clicking Sign On rather than relying on `networkidle`, which can
     resolve mid-chain.
   - The export endpoint depends on server-side search state that only gets
     populated by first loading the real tickets list route
     (`/SupportCenter/TicketCenter`, not the portal root `/`) — a bare API
     call without visiting it first returns a 500
     (`Cannot read property 'field' of undefined`).
   - `ctx.request.get(EXPORT_URL)` (Playwright's Node-based API client) fails
     with `Parse Error: Invalid header value char` against this portal's
     cookie set. Fetching via real browser navigation instead
     (`page.goto(EXPORT_URL)` wrapped in `page.expect_download()`) works
     reliably and matches what a real click does. Direct navigation to a
     download link makes `goto()` reject with a "Download is starting"
     error even on success — this must be caught and ignored, not treated as
     a failure.
   - `filterValues` must be percent-encoded (`%7B%7D`) in the URL, not passed
     as literal `{}`.
4. ~~Add a scheduled Windows Task~~ **Done** — Task Scheduler task named
   **"BNY Executive Dashboard"** runs `Run_BNY_Dashboard.bat` (fetch -> HTML ->
   PPTX -> PDF -> publish) **Mon-Fri at 7:30am**, logon mode "Interactive only"
   (runs in the user's session, no stored password; requires the machine to be
   on and the user logged in at trigger time). To view status/history or
   change the schedule:
   - GUI: `Win+R` -> `taskschd.msc` -> Task Scheduler Library -> "BNY Executive
     Dashboard" (General/Triggers/History tabs).
   - CLI: `schtasks /query /tn "BNY Executive Dashboard" /fo LIST /v`.
   - To change only the start time: `schtasks /change /tn "BNY Executive
     Dashboard" /st HH:MM` (may prompt "Enter the run as password" - safe to
     press Enter/leave blank since the task is "Interactive only" and stores
     no password). To change days/recurrence, delete and recreate with
     `schtasks /create ... /sc weekly /d MON,TUE,WED,THU,FRI /st HH:MM /f`.
5. Revisit the service-account/API-token path as a lower-maintenance long-term option.

## 8. Stage 2 — Delivery (email / SharePoint)

Goal: get the generated PDF into the hands of stakeholders each morning without
manual steps, ideally via email.

- **Outlook COM automation** (`send_dashboard_email.py`, uses `pywin32`): built
  and wired in, but **not usable on this machine** - classic desktop Outlook
  (`OUTLOOK.EXE`, exposes the `Outlook.Application` COM class) is not
  installed; only "New Outlook" / web Outlook are available here, and neither
  supports COM automation. Kept in the repo for if/when classic Outlook is
  installed - no code changes needed at that point.
- **Interim solution (current, in production)**: `publish_to_sharepoint.py`
  copies the latest PDF into the OneDrive desktop-sync folder for
  `Applications\BNY Services`
  (`C:\Users\tsmith\OneDrive - VICTORIAN FUNDS MANAGEMENT CORPORATION\Applications\BNY Services`),
  which syncs automatically to the team SharePoint site. No Graph API/app
  registration needed - relies entirely on the OneDrive desktop sync client
  already running. Wired into `Run_BNY_Dashboard.bat` as the last step.
  A Power Automate flow (trigger: file created in that SharePoint folder) or a
  "New Outlook" agent is expected to pick the file up from there and send the
  actual email - that flow is configured directly in Power Automate/Outlook,
  outside this repo.
- **Longer-term option**: Microsoft Graph API (`Mail.Send` via an Azure AD app
  registration) - works headlessly with no Outlook app required, a better fit
  for an unattended scheduled task than COM automation, but needs a one-time
  app registration + admin consent. Not yet started.
