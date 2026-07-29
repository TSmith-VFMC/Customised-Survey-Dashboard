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
3. Are there **filters** we should bake into `filterValues` (e.g. environment,
   customer) rather than exporting everything?
4. Preferred **cadence** — on demand, or a scheduled Windows Task?
5. Exact **login-page field selectors** (username / password / submit) — to be
   captured from the live page during implementation.

## 7. Rollout Plan
1. Build `fetch_latest_export.py` with `--set-credentials` and a headless
   `fetch_export()` (no changes to existing generators).
2. Capture the login-page selectors and confirm a clean headless download.
3. Manual validation over a few refresh cycles to confirm session longevity.
4. Add the batch-file wiring, then optionally a scheduled Windows Task.
5. Revisit the service-account/API-token path as a lower-maintenance long-term option.
