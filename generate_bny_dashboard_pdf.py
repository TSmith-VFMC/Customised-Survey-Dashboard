"""
generate_bny_dashboard_pdf.py
--------------------------------
Renders the HTML dashboard (generate_bny_dashboard_html.py's build_html) to a
3-page PDF using headless Chromium (Playwright), honoring the same
@media print / @page rules already baked into the HTML - so pagination and
layout match exactly what you'd get from the browser's Print -> Save as PDF.
Suitable for emailing as an attachment.

Run:
    python generate_bny_dashboard_pdf.py

Output:
    BNY_Executive_Dashboard_Services_<DDMMYYYY>.pdf written to the source dir.
"""

import os

import pandas as pd
from playwright.sync_api import sync_playwright

from generate_bny_dashboard import (
    CONFIG,
    compute_metrics,
    find_latest_source_file,
    load_cases,
    parse_asof_from_filename,
)
from generate_bny_dashboard_html import build_html


def render_pdf(html: str, out_path: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.emulate_media(media="print")
            page.pdf(path=out_path, print_background=True, prefer_css_page_size=True, scale=0.85)
        finally:
            browser.close()


def main():
    source_path = find_latest_source_file(CONFIG["source_dir"], CONFIG["source_glob"])
    asof = pd.Timestamp(parse_asof_from_filename(source_path))
    print(f"Using source file: {source_path}")
    print(f"As-of timestamp:   {asof}")

    df = load_cases(source_path)
    m = compute_metrics(df, asof)

    html = build_html(m)
    out_name = f'BNY_Executive_Dashboard_Services_{asof.strftime("%d%m%Y")}.pdf'
    out_path = os.path.join(CONFIG["source_dir"], out_name)
    render_pdf(html, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
