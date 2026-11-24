# D&AA resource allocation sync

[`sync_dna_allocation.py`](sync_dna_allocation.py) copies Data & Advanced Analytics project allocations from

- **Source:** `Team - Strategic Technology & Advanced Analytics - Team Information\D&A Resource Allocation Summaryv2.xlsx` → sheet `OLT Mgmt`

into

- **Target:** `Team - OLT - Resource Management\Operations Team_ Resource Management.xlsx` → sheet `Project Resource (new)` (the `Data & Advanced Analytics` block)

## Double-click version

`Run_DnA_Resource_Sync.bat` sits in the D&A Team Information folder next to the source workbook, with the script in an `_app` subfolder (same layout as `Run_BNY_Dashboard.bat`). It previews the changes, asks `Y/N`, then applies them. The script finds both workbooks relative to its own location, so it works for anyone who has both Teams folders synced.

After changing the script or the `.bat`, republish with:

```powershell
python resource_allocation\publish.py
```

## Config (staff names)

Staff names are kept out of this public repo. Copy [`dna_sync_config.example.json`](dna_sync_config.example.json) to `dna_sync_config.json` (git-ignored) in this folder and fill in `excluded_people` and `name_aliases`. `publish.py` copies it to `_app\` alongside the script. The script stops with an error if the config file is missing.

## Usage

```powershell
python resource_allocation\sync_dna_allocation.py            # dry run - shows the changes
python resource_allocation\sync_dna_allocation.py --apply    # write the changes
python resource_allocation\sync_dna_allocation.py --from 2027-01 --apply   # only Jan-27 onwards
```

Close the target workbook in desktop Excel before running `--apply`. A timestamped backup is written to `%LOCALAPPDATA%\DnA_Resource_Sync\backups\` before every write. It is kept local on purpose, because the target workbook also contains the OLT leave registers.

## Rules

| Source `Category` | Target `Project` |
|---|---|
| `E28 Fut Tech AA` | Future Tech |
| `E28 Modern Cloud` (all sub-projects summed) | Modern Cloud |
| `E28 Elevated Partners` | Elevated Partnerships |

- Only source rows with Team = `Data & Advanced Analytics`, excluding anyone in `excluded_people` in the config file (below).
- Months are matched by the date in the header row, from October 2026 onwards (`DEFAULT_FROM_MONTH`). A source month header that repeats the previous month (e.g. a second Dec-26) is treated as the following month.
- Where someone's name is spelt differently in the two workbooks, map it in `name_aliases` (OLT spelling → D&A summary spelling).
- Project lines that are zero/blank for every month are removed; new in-scope projects/people are added with matching formatting and merged name cells.
- Other projects in the target (e.g. Project Pinnacle, Performance Op Model) are left untouched.

The target is patched at the XML level so the rest of the shared workbook (Leave Register conditional formatting, threaded comments, data validation) is preserved exactly. Saving the workbook with openpyxl would silently drop those features.
