"""
Publish the D&AA resource allocation sync to the synced D&A Teams folder.
The .bat, script and config all go into a Resource_app folder inside
Team Information (next to the source workbook), so anyone with that folder
synced can double-click Run_DnA_Resource_Sync.bat without cloning this repo.

Run this after changing sync_dna_allocation.py or the .bat:
    python resource_allocation/publish.py
"""

import shutil
from pathlib import Path

from sync_dna_allocation import SOURCE_PATH

HERE = Path(__file__).resolve().parent
APP_FOLDER = "Resource_app"


def main() -> None:
    app_dir = SOURCE_PATH.parent / APP_FOLDER
    app_dir.mkdir(exist_ok=True)

    for name in ("Run_DnA_Resource_Sync.bat", "sync_dna_allocation.py"):
        shutil.copy2(HERE / name, app_dir / name)
        print(f"Published: {APP_FOLDER}\\{name}")
    # Local, git-ignored config with staff names (see dna_sync_config.example.json).
    config = HERE / "dna_sync_config.json"
    if config.exists():
        shutil.copy2(config, app_dir / config.name)
        print(f"Published: {APP_FOLDER}\\{config.name}")
    elif not (app_dir / config.name).exists():
        print(f"WARNING: {config.name} not found - copy the example, fill it in and republish.")
    print(f"\nDone. Published to: {app_dir}")


if __name__ == "__main__":
    main()
