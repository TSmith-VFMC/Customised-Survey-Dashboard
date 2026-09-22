"""
publish_app.py
--------------------------------
Copies the dashboard scripts from this repo into the `_app` subfolder of the
synced Teams/SharePoint folder (CONFIG["source_dir"]), so anyone who has that
folder synced can run Run_BNY_Dashboard.bat without cloning the repo.

Run this after changing any of the pipeline scripts:
    python publish_app.py
"""

import os
import shutil

from generate_bny_dashboard import CONFIG

# Scripts the .bat needs at runtime (generate_bny_dashboard.py holds the shared
# CONFIG and helpers imported by the others).
APP_SCRIPTS = (
    "generate_bny_dashboard.py",
    "generate_bny_dashboard_html.py",
    "generate_bny_dashboard_pdf.py",
    "fetch_latest_export.py",
    "archive_outputs.py",
)


def main() -> None:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(CONFIG["source_dir"], "_app")
    os.makedirs(app_dir, exist_ok=True)

    for name in APP_SCRIPTS:
        src = os.path.join(repo_dir, name)
        dst = os.path.join(app_dir, name)
        shutil.copy2(src, dst)
        print(f"Published: {name}")

    print(f"\nDone. Scripts copied to: {app_dir}")


if __name__ == "__main__":
    main()
