"""Daily cron entrypoint: pull and index any newly published Hansards.

Run directly:
    python -m scripts.sync_hansards

Or on a schedule (see the `sync` service in docker-compose.yml, which runs
this once a day) — same call path as the manual "sync now" trigger at
POST /ingest, so behavior is identical either way.
"""

from __future__ import annotations

import logging
import sys

from pipeline.sync import sync_hansards


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    result = sync_hansards()
    if result["up_to_date"]:
        print("Already up to date — nothing new to index.")
    else:
        print(
            f"Downloaded {result['documents_downloaded']} document(s), "
            f"indexed {result['chunks_written']} chunk(s)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
