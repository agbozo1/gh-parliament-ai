"""Sanity-check the column-aware PDF extraction against a real file before
committing to a full re-ingest.

Prints the old (plain) extraction and the new (column-aware) extraction for
one page side by side, so you can visually confirm the new one actually
reads correctly instead of interleaving column text. Pass --words to
instead dump the real word x-position histogram for that page, useful for
diagnosing why column detection isn't finding the actual gutter.

Usage:
  python -m scripts.compare_extraction "data/raw/20th November, 2025.pdf" [page_number]
  python -m scripts.compare_extraction "data/raw/20th November, 2025.pdf" [page_number] --words
"""

from __future__ import annotations

import sys

import pdfplumber

from pipeline.ingest import _extract_page_text


def print_word_histogram(page, buckets: int = 60) -> None:
    words = page.extract_words()
    print(f"{len(words)} word(s) on this page; page width = {page.width:.1f}pt\n")

    bucket_width = page.width / buckets
    counts = [0] * buckets
    for word in words:
        center = (word["x0"] + word["x1"]) / 2
        bucket = min(buckets - 1, max(0, int(center / bucket_width)))
        counts[bucket] += 1

    max_count = max(counts) or 1
    for i, count in enumerate(counts):
        x_start = i * bucket_width
        bar = "#" * int(count / max_count * 40)
        print(f"{x_start:6.1f}pt | {bar} {count}")

    # Also list every word's raw (x0, x1, top) so an exact gutter position
    # can be read off directly, in case the histogram is ambiguous.
    print("\nFirst 40 words (x0, x1, top, text):")
    for word in words[:40]:
        print(f"  x0={word['x0']:.1f} x1={word['x1']:.1f} top={word['top']:.1f}  {word['text']!r}")


def main(pdf_path: str, page_number: int, show_words: bool) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            print(f"PDF only has {len(pdf.pages)} page(s); pass a smaller page number.")
            return
        page = pdf.pages[page_number]

        if show_words:
            print_word_histogram(page)
            return

        print(f"=== OLD (plain extract_text) -- page {page_number} ===\n")
        print(page.extract_text() or "(empty)")

        print(f"\n\n=== NEW (column-aware) -- page {page_number} ===\n")
        print(_extract_page_text(page))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python -m scripts.compare_extraction "path/to/file.pdf" '
            "[page_number] [--words]"
        )
        sys.exit(1)
    args = [a for a in sys.argv[1:] if a != "--words"]
    words_flag = "--words" in sys.argv
    page_arg = int(args[1]) if len(args) > 1 else 0
    main(args[0], page_arg, words_flag)
