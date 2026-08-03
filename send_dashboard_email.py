"""
send_dashboard_email.py
--------------------------------
Stage 2 (email delivery): attaches the latest BNY Executive Dashboard PDF to a
new Outlook email and sends it via the desktop Outlook app (COM automation -
uses whatever mailbox you're signed into, no separate credentials needed).

Requires: Outlook desktop installed and signed in on this machine.

Run:
    python send_dashboard_email.py

Config (EMAIL_CONFIG below): recipients, subject, body.
"""

import glob
import os

import win32com.client

from generate_bny_dashboard import CONFIG

EMAIL_CONFIG = {
    # Trial: tsmith only for now. Add the rest back once we're comfortable with
    # the automated send:
    #   "DA@vfmc.vic.gov.au", "cli@vfmc.vic.gov.au",
    #   "anorton@vfmc.vic.gov.au", "nanuwar@vfmc.vic.gov.au",
    "to": [
        "tsmith@vfmc.vic.gov.au",
    ],
    "subject": "TRIAL - BNY Services Daily Start of Day Scorecard - Services cases",
    "body": (
        "Good morning,\n\n"
        "Attached is the latest BNY Managed Data Services Executive Service Health "
        "dashboard, generated automatically from the Eagle client portal export.\n\n"
        "This is a trial of the automated daily scorecard - please flag any issues.\n\n"
        "Regards"
    ),
    "pdf_glob": "BNY_Executive_Dashboard_Services_*.pdf",
}


def find_latest_pdf(source_dir: str, pattern: str) -> str:
    candidates = glob.glob(os.path.join(source_dir, pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {source_dir}")
    return max(candidates, key=os.path.getmtime)


def send_email(pdf_path: str) -> None:
    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # olMailItem
    mail.To = "; ".join(EMAIL_CONFIG["to"])
    mail.Subject = EMAIL_CONFIG["subject"]
    mail.Body = EMAIL_CONFIG["body"]
    mail.Attachments.Add(pdf_path)
    mail.Send()


def main():
    pdf_path = find_latest_pdf(CONFIG["source_dir"], EMAIL_CONFIG["pdf_glob"])
    print(f"Attaching: {pdf_path}")
    send_email(pdf_path)
    print(f"Sent to: {', '.join(EMAIL_CONFIG['to'])}")


if __name__ == "__main__":
    main()
