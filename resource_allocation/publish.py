"""
Publish the D&AA resource allocation sync to the synced D&A Teams folder
(next to the source workbook) so anyone with that folder synced can
double-click Run_DnA_Resource_Sync.bat without cloning this repo.

Run this after changing sync_dna_allocation.py or the .bat:
    python resource_allocation/publish.py
"""

import shutil
from pathlib import Path

from sync_dna_allocation import SOURCE_PATH

HERE = Path(__file__).resolve().parent


def main() -> None:
    dest = SOURCE_PATH.parent
    app_dir = dest / "_app"
    app_dir.mkdir(exist_ok=True)

    shutil.copy2(HERE / "sync_dna_allocation.py", app_dir / "sync_dna_allocation.py")
    print("Published: _app\\sync_dna_allocation.py")
    # Local, git-ignored config with staff names (see dna_sync_config.example.json).
    config = HERE / "dna_sync_config.json"
    if config.exists():
        shutil.copy2(config, app_dir / config.name)
        print(f"Published: _app\\{config.name}")
    elif not (app_dir / config.name).exists():
        print(f"WARNING: {config.name} not found - copy the example, fill it in and republish.")
    shutil.copy2(HERE / "Run_DnA_Resource_Sync.bat", dest / "Run_DnA_Resource_Sync.bat")
    print("Published: Run_DnA_Resource_Sync.bat")
    print(f"\nDone. Published to: {dest}")


if __name__ == "__main__":
    main()
