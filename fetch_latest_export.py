"""
fetch_latest_export.py
--------------------------------
Fetches the latest VFM Cases ticket export from the Eagle Investment Systems
client portal and saves it into CONFIG["source_dir"] (see
generate_bny_dashboard.py), so the dashboard generators pick it up as the
newest source file automatically.

Login is via BNY/VFMC single sign-on (PingFederate) using an email address as
the username, plain-password only (no MFA). Credentials are never stored in
this repo, in code, or passed to any script argument - they live only in
Windows Credential Manager via the `keyring` library.

One-time setup:
    python fetch_latest_export.py --set-credentials
    (prompts for your VFMC email + portal password via getpass)

Usage:
    python fetch_latest_export.py
    python fetch_latest_export.py --headed   # visible browser, for debugging login

See docs/automated-export-spec.md for the full design.
"""

import argparse
import msvcrt
import sys
from datetime import datetime
from pathlib import Path

import keyring
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from generate_bny_dashboard import CONFIG

PORTAL_LOGIN = (
    "https://myeagleapps.eagleinvsys.com/idp/startSSO.ping"
    "?PartnerSpId=XPCPSP&TargetResource=https%3A%2F%2Fxpclientportal.eagleinvsys.com%2F"
)
PORTAL_HOME = "https://xpclientportal.eagleinvsys.com/"
TICKETS_LIST_URL = "https://xpclientportal.eagleinvsys.com/SupportCenter/TicketCenter"
EXPORT_URL = (
    "https://xpclientportal.eagleinvsys.com/api/v1/tickets"
    "?export=EXCEL&sortBy=updatedAt&isDesc=1"
    "&selectedColumns=number%2Cstate%2Cenvironment%2Csid%2Cpriority%2CshortDescription"
    "%2CopenedAt%2CupdatedAt%2CopenedBy%2CcustomerNumber%2CcaseType%2CbugTrackerNumber"
    "%2Csubcategory%2CcustomerKeyword%2Ccategory&filterValues=%7B%7D"
)
LOGIN_SELECTORS = {
    "username": "#username",
    "password": "#password",
    "submit": "a.ping-button.allow",
}
AUTH_DIR = str(Path(__file__).parent / ".auth")
KEYRING_SERVICE = "eagle-xpclientportal"
KEYRING_USERNAME_KEY = "username"  # fixed lookup key storing the login email
XLSX_MAGIC = b"PK\x03\x04"  # zip/xlsx file signature


def _masked_input(prompt: str) -> str:
    """Read a password from the console, echoing '*' per keystroke so typing is visibly registered."""
    print(prompt, end="", flush=True)
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            print()
            return "".join(chars)
        if ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if ch in ("\x08", "\x7f"):  # Backspace
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)
            continue
        chars.append(ch)
        print("*", end="", flush=True)


def set_credentials() -> None:
    """Interactively prompt for and store the portal login in Windows Credential Manager."""
    username = input("VFMC login email (e.g. tsmith@vfmc.vic.gov.au): ").strip()
    password = _masked_input("Portal password: ")
    confirm = _masked_input("Re-enter portal password: ")
    if password != confirm:
        print("Passwords did not match - nothing was stored. Please try again.", file=sys.stderr)
        sys.exit(1)
    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME_KEY, username)
    keyring.set_password(KEYRING_SERVICE, username, password)
    print(f"Stored credentials for {username} in Windows Credential Manager.")


def _load_credentials() -> tuple[str, str]:
    username = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME_KEY)
    if not username:
        raise RuntimeError(
            "No stored credentials found. Run: python fetch_latest_export.py --set-credentials"
        )
    password = keyring.get_password(KEYRING_SERVICE, username)
    if not password:
        raise RuntimeError(
            f"No stored password found for {username}. Run: python fetch_latest_export.py --set-credentials"
        )
    return username, password


def _export_filename(now: datetime) -> str:
    hour_12 = now.strftime("%I").lstrip("0") or "12"
    return f"VFM_Cases_{now.strftime('%m-%d-%Y')}_{hour_12}_{now.strftime('%M_%p')}.xlsx"


def _login(page, username: str, password: str, debug: bool = False) -> None:
    page.goto(PORTAL_LOGIN)
    page.fill(LOGIN_SELECTORS["username"], username)
    page.fill(LOGIN_SELECTORS["password"], password)
    page.click(LOGIN_SELECTORS["submit"])
    # The SAML flow hops through several intermediate redirects; wait for the
    # actual portal domain rather than "networkidle", which can resolve early
    # during a brief pause mid-chain.
    page.wait_for_url("**xpclientportal.eagleinvsys.com**", timeout=20000)
    page.wait_for_load_state("networkidle")
    if debug:
        print(f"[debug] post-login URL: {page.url}", file=sys.stderr)
        cookie_domains = sorted({c["domain"] for c in page.context.cookies()})
        print(f"[debug] cookie domains: {cookie_domains}", file=sys.stderr)
        shot_path = Path(AUTH_DIR) / "debug-post-login.png"
        page.screenshot(path=str(shot_path))
        print(f"[debug] screenshot saved to {shot_path}", file=sys.stderr)


def _visit_tickets_list(page, debug: bool = False) -> None:
    """Load the tickets list so the portal populates the server-side search
    state the export endpoint reads from (a bare API call without this returns
    a 500 'Cannot read property field of undefined')."""
    page.goto(TICKETS_LIST_URL, wait_until="networkidle")
    if debug:
        print(f"[debug] tickets list URL: {page.url}", file=sys.stderr)


def _download_export(page, debug: bool = False) -> bytes | None:
    """Navigate to the export URL and capture the resulting file download,
    exactly as a real click on the export icon would. Returns None if the URL
    didn't trigger a download (e.g. it rendered an error page instead)."""
    try:
        with page.expect_download(timeout=15000) as download_info:
            try:
                page.goto(EXPORT_URL)
            except PlaywrightError as exc:
                # Direct navigation to a download link makes goto() reject
                # with this message even though the download proceeds fine.
                if "Download is starting" not in str(exc):
                    raise
        download = download_info.value
        tmp_path = Path(AUTH_DIR) / "_export_tmp.xlsx"
        download.save_as(str(tmp_path))
        data = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
        return data
    except PlaywrightTimeoutError:
        if debug:
            print(f"[debug] export did not trigger a download - landed on: {page.url}", file=sys.stderr)
            print(f"[debug] page content (first 500 chars): {page.content()[:500]!r}", file=sys.stderr)
        return None


def fetch_export(headless: bool = True, debug: bool = False) -> Path:
    username, password = _load_credentials()
    Path(AUTH_DIR).mkdir(exist_ok=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(AUTH_DIR, headless=headless)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            _visit_tickets_list(page, debug=debug)
            body = _download_export(page, debug=debug)
            if body is None or not body.startswith(XLSX_MAGIC):
                _login(page, username, password, debug=debug)
                _visit_tickets_list(page, debug=debug)
                body = _download_export(page, debug=debug)

            if body is None or not body.startswith(XLSX_MAGIC):
                raise RuntimeError(
                    "Export request did not return a spreadsheet - login may have failed "
                    "or the session expired."
                )

            out_path = Path(CONFIG["source_dir"]) / _export_filename(datetime.now())
            out_path.write_bytes(body)
            return out_path
        finally:
            ctx.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set-credentials",
        action="store_true",
        help="Prompt for and store the portal login in Windows Credential Manager, then exit.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run the browser headed (visible) - useful for debugging login issues.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (post-login URL, cookie domains, response status/body) - no credentials are ever printed.",
    )
    args = parser.parse_args()

    if args.set_credentials:
        set_credentials()
        return

    try:
        path = fetch_export(headless=not args.headed, debug=args.debug)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
