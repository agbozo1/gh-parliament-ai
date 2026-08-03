"""Sanity-check the column-aware PDF extraction against a real file before
committing to a full re-ingest.

Prints the old (plain) extraction and the new (column-aware) extraction for
one page side by side, so you can visually confirm the new one actually
reads correctly instead of interleaving column text.

Usage: python -m scripts.compare_extraction "data/raw/20th November, 2025.pdf" [page_number]
"""

from __future__ import annotations

import sys

import pdfplumber

from pipeline.ingest import _extract_page_text


def main(pdf_path: str, page_number: int) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            print(f"PDF only has {len(pdf.pages)} page(s); pass a smaller page number.")
            return
        page = pdf.pages[page_number]

        print(f"=== OLD (plain extract_text) -- page {page_number} ===\n")
        print(page.extract_text() or "(empty)")

        print(f"\n\n=== NEW (column-aware) -- page {page_number} ===\n")
        print(_extract_page_text(page))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.compare_extraction "path/to/file.pdf" [page_number]')
        sys.exit(1)
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(sys.argv[1], page_arg)
