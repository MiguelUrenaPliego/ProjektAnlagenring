#!/usr/bin/env python3
"""Render presentation_base.pdf pages to JPG images for use in presentation.md.

Mobile browsers (iOS Safari, Android Chrome) don't have a built-in PDF
viewer plugin for <iframe> embeds, so the live-PDF-in-an-iframe approach
used on desktop just shows a fallback "open presentation_base.pdf" link
there. Pre-rendering the pages we actually use as JPGs and embedding those
with <img> instead works on every device.

Run this whenever presentation_base.pdf changes:

    python3 marp/render_pdf_pages.py

It (re)renders every page referenced in presentation.md into
marp/pdf_pages/page-N.jpg. Requires poppler-utils (`pdftoppm`) to be
installed (apt install poppler-utils / brew install poppler).
"""
import re
import subprocess
import sys
from pathlib import Path

MARP_DIR = Path(__file__).resolve().parent
PDF_PATH = MARP_DIR / "presentation_base.pdf"
OUT_DIR = MARP_DIR / "pdf_pages"
PRESENTATION_MD = MARP_DIR / "presentation.md"
DPI = 150
JPEG_QUALITY = 85


def used_page_numbers():
    text = PRESENTATION_MD.read_text(encoding="utf-8")
    pages = {int(n) for n in re.findall(r'data-pdf-page="(\d+)"', text)}
    if not pages:
        sys.exit('No \'data-pdf-page="N"\' references found in presentation.md')
    return sorted(pages)


def render_page(page: int) -> None:
    out_prefix = OUT_DIR / f"page-{page}"
    subprocess.run(
        [
            "pdftoppm",
            "-jpeg",
            "-jpegopt", f"quality={JPEG_QUALITY}",
            "-r", str(DPI),
            "-f", str(page),
            "-l", str(page),
            str(PDF_PATH),
            str(out_prefix),
        ],
        check=True,
    )
    # pdftoppm appends "-<page>" to the prefix, zero-padded to the digit
    # width of the source PDF's page count (e.g. "-02" for a 17-page PDF)
    # — glob for it instead of guessing the padding, then drop the suffix.
    [rendered] = OUT_DIR.glob(f"{out_prefix.name}-*.jpg")
    target = out_prefix.with_suffix(".jpg")
    rendered.rename(target)
    print(f"  page {page} -> {target.relative_to(MARP_DIR)}")


def main():
    if not PDF_PATH.exists():
        sys.exit(f"Not found: {PDF_PATH}")
    OUT_DIR.mkdir(exist_ok=True)
    pages = used_page_numbers()
    print(f"Rendering {len(pages)} page(s) from {PDF_PATH.name} at {DPI} DPI...")
    for page in pages:
        render_page(page)
    print("Done.")


if __name__ == "__main__":
    main()
