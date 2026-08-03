"""
publish_to_sharepoint.py
--------------------------------
Interim Stage 2 delivery mechanism: copies the latest generated dashboard PDF
into the user's OneDrive-synced "BNY Services" folder, which syncs
automatically to the team SharePoint site:
https://vfmcorp-my.sharepoint.com/personal/tsmith_vfmc_vic_gov_au/Documents/Applications/BNY%20Services

No Graph API / app registration needed - this just relies on the OneDrive
desktop sync client already running on this machine. Once the file lands on
SharePoint, a Power Automate flow (or "New Outlook" agent) can watch the
folder and send the email.

Run:
    python publish_to_sharepoint.py
"""

import glob
import os
import shutil

from generate_bny_dashboard import CONFIG

PUBLISH_CONFIG = {
    "pdf_glob": "BNY_Executive_Dashboard_Services_*.pdf",
    "target_dir": (
        r"C:\Users\tsmith\OneDrive - VICTORIAN FUNDS MANAGEMENT CORPORATION"
        r"\Applications\BNY Services"
    ),
}


def find_latest_pdf(source_dir: str, pattern: str) -> str:
    candidates = glob.glob(os.path.join(source_dir, pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {source_dir}")
    return max(candidates, key=os.path.getmtime)


def main():
    pdf_path = find_latest_pdf(CONFIG["source_dir"], PUBLISH_CONFIG["pdf_glob"])
    target_dir = PUBLISH_CONFIG["target_dir"]
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"OneDrive target folder not found: {target_dir}\n"
            "Check the OneDrive sync client is running and the folder name/path is correct."
        )
    dest_path = os.path.join(target_dir, os.path.basename(pdf_path))
    shutil.copy2(pdf_path, dest_path)
    print(f"Published: {dest_path}")


if __name__ == "__main__":
    main()
