"""
Sync Data & Advanced Analytics project allocations from the D&A Resource
Allocation Summary ('OLT Mgmt' sheet) into the OLT Operations Team Resource
Management workbook ('Project Resource (new)' sheet).

Only the Future Tech, Modern Cloud and Elevated Partnerships projects are
synced, from October 2026 onwards (override with --from YYYY-MM).

The target workbook is patched at the XML level (only the one worksheet part
is rewritten) so conditional formatting, threaded comments, data validation
extensions etc. elsewhere in the shared workbook are preserved byte-for-byte.
openpyxl would silently drop those on save.

Usage:
    python resource_allocation/sync_dna_allocation.py            # dry run
    python resource_allocation/sync_dna_allocation.py --apply    # write
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from openpyxl import load_workbook

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SOURCE_FOLDER = "Team - Strategic Technology & Advanced Analytics - Team Information"
SOURCE_NAME = "D&A Resource Allocation Summaryv2.xlsx"
TARGET_FOLDER = "Team - OLT - Resource Management"
TARGET_NAME = "Operations Team_ Resource Management.xlsx"


def _sync_root() -> Path:
    """Folder holding the synced Teams/SharePoint libraries (any user).

    When run from the published `_app` folder the script sits in
    <root>\\<SOURCE_FOLDER>\\_app, so the root is found relative to the script.
    """
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / SOURCE_NAME).exists() and d.name == SOURCE_FOLDER:
            return d.parent
    return Path.home() / "VICTORIAN FUNDS MANAGEMENT CORPORATION"


_ROOT = _sync_root()
SOURCE_PATH = _ROOT / SOURCE_FOLDER / SOURCE_NAME
TARGET_PATH = _ROOT / TARGET_FOLDER / TARGET_NAME
SOURCE_SHEET = "OLT Mgmt"
TARGET_SHEET = "Project Resource (new)"
# Kept on the local machine (not the synced folder) - the target workbook also
# holds the OLT leave registers, which must not be copied into the D&A site.
BACKUP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DnA_Resource_Sync" / "backups"
EXIT_CHANGES_PENDING = 10  # dry run found changes (used by the .bat)

DEFAULT_FROM_MONTH = (2026, 10)  # first month column synced (override with --from)

TEAM = "Data & Advanced Analytics"

# Source 'Category' -> target 'Project'. Multiple source rows that map to the
# same target project for a person are summed.
CATEGORY_TO_PROJECT = {
    "E28 Fut Tech AA": "Future Tech",
    "E28 Modern Cloud": "Modern Cloud",
    "E28 Elevated Partners": "Elevated Partnerships",
}
IN_SCOPE_PROJECTS = set(CATEGORY_TO_PROJECT.values())

# People-specific settings live in a local, git-ignored JSON file next to this
# script (the repo is public). See dna_sync_config.example.json.
CONFIG_PATH = Path(__file__).resolve().parent / "dna_sync_config.json"


def _load_people_config() -> tuple[set[str], dict[str, str]]:
    if not CONFIG_PATH.exists():
        return set(), {}
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    excluded = {" ".join(n.split()) for n in cfg.get("excluded_people", [])}
    # Target spelling -> source spelling.
    aliases = {" ".join(k.split()): " ".join(v.split()) for k, v in cfg.get("name_aliases", {}).items()}
    return excluded, aliases


EXCLUDED_PEOPLE, NAME_ALIASES = _load_people_config()

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
EXCEL_EPOCH = datetime(1899, 12, 30)


def month_key(d: date | datetime) -> tuple[int, int]:
    return (d.year, d.month)


def next_month(key: tuple[int, int]) -> tuple[int, int]:
    y, m = key
    return (y + 1, 1) if m == 12 else (y, m + 1)


def fmt_month(key: tuple[int, int]) -> str:
    return date(key[0], key[1], 1).strftime("%b-%y")


def canonical_name(name: str) -> str:
    name = " ".join(str(name).split())
    return NAME_ALIASES.get(name, name)


def norm(text) -> str:
    return " ".join(str(text).split()) if text is not None else ""


def col_to_idx(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def fmt_num(v: float | None) -> str:
    if v is None:
        return ""
    v = round(float(v), 6)
    return str(int(v)) if v.is_integer() else repr(v)


def is_zero(v) -> bool:
    return v is None or abs(float(v)) < 1e-9


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------
def read_source(path: Path) -> tuple[dict, list[tuple[int, int]]]:
    """Return ({(person, project): {month_key: value}}, months)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[SOURCE_SHEET]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    hdr_idx = next(
        i for i, r in enumerate(rows) if r and norm(r[0]).lower() == "resource"
    )
    header = rows[hdr_idx]
    col = {norm(v).lower(): i for i, v in enumerate(header) if isinstance(v, str)}
    c_res, c_team, c_cat = col["resource"], col["team"], col["category"]

    # Month columns; a header that does not advance past the previous month is
    # treated as a typo for the next month (e.g. Dec-26 repeated -> Jan-27).
    month_cols: dict[int, tuple[int, int]] = {}
    prev = None
    for i, v in enumerate(header):
        if not isinstance(v, (datetime, date)):
            continue
        key = month_key(v)
        if prev is not None and key <= prev:
            fixed = next_month(prev)
            print(
                f"  NOTE: source column {i + 1} header {fmt_month(key)} "
                f"repeats/precedes {fmt_month(prev)}; treating as {fmt_month(fixed)}"
            )
            key = fixed
        month_cols[i] = key
        prev = key

    data: dict[tuple[str, str], dict[tuple[int, int], float | None]] = {}
    for r in rows[hdr_idx + 1 :]:
        if not r or r[c_res] is None:
            continue
        person = canonical_name(r[c_res])
        if norm(r[c_team]) != TEAM or person in EXCLUDED_PEOPLE:
            continue
        project = CATEGORY_TO_PROJECT.get(norm(r[c_cat]))
        if project is None:
            continue
        bucket = data.setdefault((person, project), {})
        for ci, mk in month_cols.items():
            v = r[ci] if ci < len(r) else None
            if isinstance(v, (int, float)):
                bucket[mk] = (bucket.get(mk) or 0) + float(v)
            else:
                bucket.setdefault(mk, None)
    return data, list(month_cols.values())


# --------------------------------------------------------------------------
# Target (raw XML)
# --------------------------------------------------------------------------
ROW_RE = re.compile(r'<row r="(\d+)"[^>]*?(?:/>|>.*?</row>)', re.S)
CELL_RE = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', re.S)


@dataclass
class Cell:
    col: str
    attrs: str  # attribute text after r="..", e.g. ' s="257" t="s"'
    inner: str | None

    @property
    def style(self) -> str:
        m = re.search(r'\bs="(\d+)"', self.attrs)
        return f' s="{m.group(1)}"' if m else ""

    def xml(self, row: int) -> str:
        if self.inner is None:
            return f'<c r="{self.col}{row}"{self.attrs}/>'
        return f'<c r="{self.col}{row}"{self.attrs}>{self.inner}</c>'


@dataclass
class Row:
    num: int
    open_tag: str  # original '<row r="N" ...>' tag
    cells: dict[str, Cell] = field(default_factory=dict)

    def xml(self, num: int) -> str:
        tag = re.sub(r'<row r="\d+"', f'<row r="{num}"', self.open_tag, 1)
        tag = tag.rstrip("/>").rstrip() + ">"
        body = "".join(
            c.xml(num) for c in sorted(self.cells.values(), key=lambda c: col_to_idx(c.col))
        )
        return f"{tag}{body}</row>"


def parse_row(xml: str) -> Row:
    num = int(re.match(r'<row r="(\d+)"', xml).group(1))
    open_tag = re.match(r"<row[^>]*?/?>", xml).group(0)
    row = Row(num, open_tag)
    for m in CELL_RE.finditer(xml):
        row.cells[m.group(1)] = Cell(m.group(1), m.group(3), m.group(4))
    return row


class SharedStrings:
    def __init__(self, xml: str):
        self.items = []
        for si in re.findall(r"<si>(.*?)</si>", xml, re.S):
            self.items.append(
                html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
            )
        self.index = {}
        for i, s in enumerate(self.items):
            self.index.setdefault(s, i)


def cell_text(cell: Cell | None, sst: SharedStrings) -> str:
    if cell is None or cell.inner is None:
        return ""
    if 't="s"' in cell.attrs:
        m = re.search(r"<v>(\d+)</v>", cell.inner)
        return norm(sst.items[int(m.group(1))]) if m else ""
    if 't="inlineStr"' in cell.attrs:
        return norm(html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", cell.inner))))
    m = re.search(r"<v>(.*?)</v>", cell.inner)
    return norm(m.group(1)) if m else ""


def cell_number(cell: Cell | None) -> float | None:
    if cell is None or cell.inner is None or "t=" in cell.attrs:
        return None
    m = re.search(r"<v>(.*?)</v>", cell.inner)
    return float(m.group(1)) if m else None


def set_text(cell: Cell, text: str, sst: SharedStrings) -> None:
    style = cell.style
    if not text:
        cell.attrs, cell.inner = style, None
    elif text in sst.index:
        cell.attrs, cell.inner = f'{style} t="s"', f"<v>{sst.index[text]}</v>"
    else:
        cell.attrs = f'{style} t="inlineStr"'
        cell.inner = f"<is><t>{html.escape(text, quote=False)}</t></is>"


def set_number(cell: Cell, value: float | None) -> None:
    style = cell.style
    if value is None:
        cell.attrs, cell.inner = style, None
    else:
        cell.attrs, cell.inner = style, f"<v>{fmt_num(value)}</v>"


def resolve_sheet_part(z: zipfile.ZipFile, sheet_name: str) -> str:
    wb = z.read("xl/workbook.xml").decode("utf-8")
    m = re.search(
        rf'<sheet [^>]*name="{re.escape(html.escape(sheet_name, quote=True))}"[^>]*r:id="(rId\d+)"', wb
    )
    rid = m.group(1)
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    target = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', rels) or re.search(
        rf'Target="([^"]+)"[^>]*Id="{rid}"', rels
    )
    t = target.group(1).lstrip("/")
    return t if t.startswith("xl/") else "xl/" + t


def sheet_index(z: zipfile.ZipFile, sheet_name: str) -> int:
    wb = z.read("xl/workbook.xml").decode("utf-8")
    names = re.findall(r'<sheet [^>]*name="([^"]+)"', wb)
    return [html.unescape(n) for n in names].index(sheet_name)


# --- row shifting for references below the D&AA block -----------------------
REF_RE = re.compile(r"(\$?)([A-Z]{1,3})(\$?)(\d+)")


def shift_refs(text: str, after_row: int, delta: int) -> str:
    """Shift every A1 row number > after_row by delta."""
    if delta == 0:
        return text

    def repl(m):
        r = int(m.group(4))
        if r > after_row:
            r += delta
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{r}"

    return REF_RE.sub(repl, text)


def shift_row_xml(xml: str, delta: int) -> str:
    xml = re.sub(r'^<row r="(\d+)"', lambda m: f'<row r="{int(m.group(1)) + delta}"', xml)
    return re.sub(
        r'<c r="([A-Z]+)(\d+)"', lambda m: f'<c r="{m.group(1)}{int(m.group(2)) + delta}"', xml
    )


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------
@dataclass
class Line:
    row: Row  # existing row, or a styled clone for new lines
    person: str  # canonical name of the person group
    project: str
    is_new: bool = False


def _vals_str(vals, month_cols) -> str:
    return " ".join(f"{fmt_month(month_cols[c])}={fmt_num(v) or '-'}" for c, v in vals.items())


def build_plan(sheet_xml: str, sst: SharedStrings, source: dict, src_months, from_key):
    parsed = {int(m.group(1)): parse_row(m.group(0)) for m in ROW_RE.finditer(sheet_xml)}

    hdr = next(r for r in parsed.values() if cell_text(r.cells.get("B"), sst).lower() == "resource")
    month_cols: dict[str, tuple[int, int]] = {}
    for col, c in hdr.cells.items():
        v = cell_number(c)
        if col_to_idx(col) > col_to_idx("D") and v is not None and v > 30000:
            month_cols[col] = month_key(EXCEL_EPOCH + timedelta(days=v))

    update_cols = {
        col: mk for col, mk in month_cols.items() if mk >= from_key and mk in src_months
    }
    for col, mk in month_cols.items():
        if mk >= from_key and mk not in src_months:
            print(f"  WARN: target month {fmt_month(mk)} not in source; left unchanged")

    # D&AA block: first row with Team = TEAM through the last contiguous row
    # before a row belonging to another team.
    nums = sorted(n for n in parsed if n > hdr.num)
    start = next(n for n in nums if cell_text(parsed[n].cells.get("C"), sst) == TEAM)
    end = start
    for n in nums:
        if n <= start:
            continue
        team = cell_text(parsed[n].cells.get("C"), sst)
        if (team and team != TEAM) or n != end + 1:
            break
        end = n

    groups: list[tuple[str, str, list[Line]]] = []  # (canonical, display, lines)
    for n in range(start, end + 1):
        r = parsed[n]
        name = cell_text(r.cells.get("B"), sst)
        if name:
            groups.append((canonical_name(name), name, []))
        groups[-1][2].append(Line(r, groups[-1][0], cell_text(r.cells.get("D"), sst)))

    first_tpl = parsed[start]
    cont_tpl = next(
        (parsed[n] for n in range(start, end + 1) if not cell_text(parsed[n].cells.get("B"), sst)),
        first_tpl,
    )

    def new_line(person: str, project: str, vals: dict) -> Line:
        row = Row(0, cont_tpl.open_tag, {k: Cell(c.col, c.style, None) for k, c in cont_tpl.cells.items()})
        for col, v in vals.items():
            set_number(row.cells.setdefault(col, Cell(col, "", None)), v)
        return Line(row, person, project, is_new=True)

    def values_for(person, project, line: Line | None) -> dict:
        """Final {col: value}: source for synced months, existing value otherwise."""
        src = source.get((person, project), {})
        return {
            col: src.get(update_cols[col]) if col in update_cols
            else (cell_number(line.row.cells.get(col)) if line else None)
            for col in month_cols
        }

    changes: list[str] = []
    seen_people = set()
    out_groups: list[tuple[str, str, list[Line]]] = []

    for canon, display, lines in groups:
        seen_people.add(canon)
        kept: list[Line] = []
        present = set()
        for ln in lines:
            if ln.project not in IN_SCOPE_PROJECTS or canon in EXCLUDED_PEOPLE:
                kept.append(ln)
                continue
            present.add(ln.project)
            vals = values_for(canon, ln.project, ln)
            if all(is_zero(v) for v in vals.values()):
                changes.append(f"  REMOVE  {display:<22} {ln.project} (now zero)")
                continue
            for col, v in vals.items():
                old = cell_number(ln.row.cells.get(col))
                if col in update_cols and fmt_num(old) != fmt_num(v):
                    changes.append(
                        f"  UPDATE  {display:<22} {ln.project:<22} {fmt_month(month_cols[col])}: "
                        f"{fmt_num(old) or '-':>5} -> {fmt_num(v) or '-'}"
                    )
                    set_number(ln.row.cells.setdefault(col, Cell(col, "", None)), v)
            kept.append(ln)
        if canon not in EXCLUDED_PEOPLE:
            for proj in sorted(IN_SCOPE_PROJECTS - present):
                if (canon, proj) not in source:
                    continue
                vals = values_for(canon, proj, None)
                if all(is_zero(v) for v in vals.values()):
                    continue
                kept.append(new_line(canon, proj, vals))
                changes.append(f"  ADD     {display:<22} {proj:<22} " + _vals_str(vals, month_cols))
        if not kept:
            # Keep the person on the sheet with an empty project line.
            ln = lines[0]
            set_text(ln.row.cells.setdefault("D", Cell("D", "", None)), "", sst)
            for col in month_cols:
                set_number(ln.row.cells.setdefault(col, Cell(col, "", None)), None)
            kept.append(ln)
        out_groups.append((canon, display, kept))

    # People in the source who are not yet on the target sheet.
    for person in sorted({p for p, _ in source} - seen_people - EXCLUDED_PEOPLE):
        lines = []
        for proj in sorted(IN_SCOPE_PROJECTS):
            if (person, proj) not in source:
                continue
            vals = values_for(person, proj, None)
            if all(is_zero(v) for v in vals.values()):
                continue
            lines.append(new_line(person, proj, vals))
            changes.append(
                f"  ADD     {person:<22} {proj:<22} " + _vals_str(vals, month_cols) + "  (new person)"
            )
        if lines:
            out_groups.append((person, person, lines))

    # Lay the block back out from `start`.
    out_rows: list[Row] = []
    merges: list[str] = []
    r = start
    for _canon, display, lines in out_groups:
        first = r
        for i, ln in enumerate(lines):
            row = ln.row
            if i == 0:
                for col in ("B", "C"):
                    if col not in row.cells or (ln.is_new and not row.cells[col].style):
                        row.cells[col] = Cell(col, first_tpl.cells[col].style, None)
                if ln.is_new or not cell_text(row.cells.get("B"), sst):
                    set_text(row.cells["B"], display, sst)
                    set_text(row.cells["C"], TEAM, sst)
            else:
                for col in ("B", "C"):
                    if col in row.cells and row.cells[col].inner is not None:
                        set_text(row.cells[col], "", sst)
            if ln.is_new:
                set_text(row.cells.setdefault("D", Cell("D", "", None)), ln.project, sst)
            out_rows.append(row)
            r += 1
        if r - first > 1:
            merges += [f"B{first}:B{r - 1}", f"C{first}:C{r - 1}"]

    return {
        "start": start,
        "end": end,
        "rows": out_rows,
        "merges": merges,
        "delta": len(out_rows) - (end - start + 1),
        "changes": changes,
        "update_months": sorted(update_cols.values()),
    }


def render_sheet(sheet_xml: str, plan) -> str:
    start, end, delta = plan["start"], plan["end"], plan["delta"]

    def row_repl(m):
        n = int(m.group(1))
        if n < start:
            return m.group(0)
        if n <= end:
            return "\x00BLOCK\x00" if n == start else ""
        return shift_row_xml(m.group(0), delta) if delta else m.group(0)

    body = ROW_RE.sub(row_repl, sheet_xml)
    block = "".join(row.xml(start + i) for i, row in enumerate(plan["rows"]))
    body = body.replace("\x00BLOCK\x00", block, 1)

    def merges_repl(m):
        keep = []
        for ref in re.findall(r'<mergeCell ref="([^"]+)"/>', m.group(0)):
            rows_ = [int(x) for x in re.findall(r"\d+", ref)]
            if min(rows_) >= start and max(rows_) <= end:
                continue
            keep.append(shift_refs(ref, end, delta))
        keep += plan["merges"]
        cells = "".join(f'<mergeCell ref="{x}"/>' for x in keep)
        return f'<mergeCells count="{len(keep)}">{cells}</mergeCells>'

    body = re.sub(r"<mergeCells[^>]*>.*?</mergeCells>", merges_repl, body, flags=re.S)

    if delta:
        for pat in (
            r'(<dimension ref=")([^"]+)',
            r'(<autoFilter ref=")([^"]+)',
            r"(<xm:sqref>)([^<]+)",
            r'(<dataValidation [^>]*sqref=")([^"]+)',
            r'(<conditionalFormatting [^>]*sqref=")([^"]+)',
        ):
            body = re.sub(pat, lambda m: m.group(1) + shift_refs(m.group(2), end, delta), body)
    return body


def render_workbook_xml(wb_xml: str, sheet_idx: int, end: int, delta: int) -> str:
    if not delta:
        return wb_xml
    pat = rf'(<definedName [^>]*localSheetId="{sheet_idx}"[^>]*>)([^<]*)(</definedName>)'
    return re.sub(pat, lambda m: m.group(1) + shift_refs(m.group(2), end, delta) + m.group(3), wb_xml)


def write_zip(src: Path, dst: Path, replacements: dict[str, bytes]) -> None:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = replacements.get(info.filename, zin.read(info.filename))
            zout.writestr(info, data, compress_type=info.compress_type)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument(
        "--from",
        dest="from_month",
        help="first month to sync, YYYY-MM (default: %d-%02d)" % DEFAULT_FROM_MONTH,
    )
    ap.add_argument("--source", type=Path, default=SOURCE_PATH)
    ap.add_argument("--target", type=Path, default=TARGET_PATH)
    args = ap.parse_args(argv)

    from_key = (
        tuple(int(x) for x in args.from_month.split("-"))
        if args.from_month
        else DEFAULT_FROM_MONTH
    )

    print(f"Source: {args.source}\nTarget: {args.target}")
    if not CONFIG_PATH.exists():
        print(
            f"\nERROR: config file not found: {CONFIG_PATH}\n"
            "Copy dna_sync_config.example.json to dna_sync_config.json and fill it in."
        )
        return 1
    print(f"Excluded people: {', '.join(sorted(EXCLUDED_PEOPLE)) or '(none)'}")
    for p in (args.source, args.target):
        if not p.exists():
            print(f"\nERROR: file not found: {p}\nMake sure both Teams folders are synced to this PC.")
            return 1
    source, src_months = read_source(args.source)

    with zipfile.ZipFile(args.target) as z:
        part = resolve_sheet_part(z, TARGET_SHEET)
        sheet_xml = z.read(part).decode("utf-8")
        wb_xml = z.read("xl/workbook.xml").decode("utf-8")
        sst = SharedStrings(z.read("xl/sharedStrings.xml").decode("utf-8"))
        s_idx = sheet_index(z, TARGET_SHEET)

    plan = build_plan(sheet_xml, sst, source, set(src_months), from_key)
    print(f"Months synced: {', '.join(fmt_month(m) for m in plan['update_months']) or '(none)'}")
    print(
        f"D&AA block: rows {plan['start']}-{plan['end']}"
        + (f" -> {plan['start']}-{plan['end'] + plan['delta']}" if plan["delta"] else "")
    )

    if not plan["changes"]:
        print("No changes - target is already up to date.")
        return 0
    print("Changes:")
    print("\n".join(plan["changes"]))

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write these changes.")
        return EXIT_CHANGES_PENDING

    lock = args.target.with_name("~$" + args.target.name)
    if lock.exists():
        print(f"\nERROR: {args.target.name} is open in Excel ({lock.name} exists). Close it and retry.")
        return 2

    new_sheet = render_sheet(sheet_xml, plan)
    new_wb = render_workbook_xml(wb_xml, s_idx, plan["end"], plan["delta"])

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{args.target.stem}_{datetime.now():%Y%m%d_%H%M%S}{args.target.suffix}"
    shutil.copy2(args.target, backup)

    fd, tmp = tempfile.mkstemp(suffix=".xlsx", dir=BACKUP_DIR)
    os.close(fd)
    try:
        write_zip(
            args.target,
            Path(tmp),
            {part: new_sheet.encode("utf-8"), "xl/workbook.xml": new_wb.encode("utf-8")},
        )
        with zipfile.ZipFile(tmp) as zt:
            if zt.testzip() is not None:
                raise RuntimeError("written workbook failed zip integrity check")
        shutil.copyfile(tmp, args.target)
    finally:
        os.remove(tmp)
    print(f"\nUpdated {args.target.name} (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
