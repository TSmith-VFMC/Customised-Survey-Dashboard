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
import getpass
import sys
from datetime import datetime
from pathlib import Path

import keyring
from playwright.sync_api import sync_playwright

from generate_bny_dashboard import CONFIG

PORTAL_LOGIN = (
    "https://myeagleapps.eagleinvsys.com/idp/startSSO.ping"
    "?PartnerSpId=XPCPSP&TargetResource=https%3A%2F%2Fxpclientportal.eagleinvsys.com%2F"
)
EXPORT_URL = (
    "https://xpclientportal.eagleinvsys.com/api/v1/tickets"
    "?export=EXCEL&sortBy=updatedAt&isDesc=1"
    "&selectedColumns=number,state,environment,sid,priority,shortDescription,"
    "openedAt,updatedAt,openedBy,customerNumber,caseType,bugTrackerNumber,"
    "subcategory,customerKeyword,category,issueType&filterValues={}"
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


def set_credentials() -> None:
    """Interactively prompt for and store the portal login in Windows Credential Manager."""
    username = input("VFMC login email (e.g. tsmith@vfmc.vic.gov.au): ").strip()
    password = getpass.getpass("Portal password: ")
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


def _login(page, username: str, password: str) -> None:
    page.goto(PORTAL_LOGIN)
    page.fill(LOGIN_SELECTORS["username"], username)
    page.fill(LOGIN_SELECTORS["password"], password)
    with page.expect_navigation(wait_until="networkidle"):
        page.click(LOGIN_SELECTORS["submit"])


def fetch_export(headless: bool = True) -> Path:
    username, password = _load_credentials()
    Path(AUTH_DIR).mkdir(exist_ok=True)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(AUTH_DIR, headless=headless)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            resp = ctx.request.get(EXPORT_URL)
            body = resp.body()
            if not resp.ok or not body.startswith(XLSX_MAGIC):
                _login(page, username, password)
                resp = ctx.request.get(EXPORT_URL)
                body = resp.body()

            if not resp.ok or not body.startswith(XLSX_MAGIC):
                raise RuntimeError(
                    "Export request did not return a spreadsheet - login may have failed "
                    f"or the session expired (status {resp.status})."
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
    args = parser.parse_args()

    if args.set_credentials:
        set_credentials()
        return

    try:
        path = fetch_export(headless=not args.headed)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
