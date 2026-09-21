"""
archive_outputs.py
--------------------------------
Moves the dated PDF reports and source XLS exports out of the main VFMC Daily
Scorecard folder into its Archive subfolder, leaving the central
"BNY_Executive_Dashboard_Latest.html" as the only report in the top-level
folder. Run as the final step of the pipeline (after HTML + PDF generation).

Run:
    python archive_outputs.py
"""

import glob
import os
import shutil

from generate_bny_dashboard import CONFIG

ARCHIVE_PATTERNS = ("BNY_Executive_Dashboard_Services_*.pdf", "VFM_Cases_*.xlsx")


def main():
    source_dir = CONFIG["source_dir"]
    archive_dir = os.path.join(source_dir, "Archive")
    os.makedirs(archive_dir, exist_ok=True)

    moved = 0
    for pattern in ARCHIVE_PATTERNS:
        for path in glob.glob(os.path.join(source_dir, pattern)):
            dest = os.path.join(archive_dir, os.path.basename(path))
            try:
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(path, dest)
                print(f"Archived: {dest}")
                moved += 1
            except (PermissionError, OSError) as exc:
                print(f"Skipped (in use?): {os.path.basename(path)} - {exc}")

    if moved == 0:
        print("Nothing to archive.")


if __name__ == "__main__":
    main()
