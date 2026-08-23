"""Merge per-process metric CSVs into one file.

Evaluation runs are parallelised one process per (controller, scenario, seed), each
writing its own CSV, because several processes appending to a single file can
interleave partial lines. This script stitches the shards back together.

Usage:
    python src/merge_metrics.py --pattern "outputs/tmp_marl/*.csv" --out outputs/marl_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def merge(pattern: str, out_path: str, dedupe: bool = True) -> int:
    """Merge every CSV matching ``pattern`` into ``out_path``.

    Args:
        pattern: glob for the shard CSVs.
        out_path: destination CSV.
        dedupe: drop exact duplicate rows, which appear when a run is repeated.

    Returns:
        Number of rows written.

    Raises:
        FileNotFoundError: if the glob matches nothing.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    seen: set[tuple] = set()

    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                key = tuple(sorted(row.items()))
                if dedupe and key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    def sort_key(row: dict[str, str]) -> tuple:
        try:
            seed = int(row.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0
        return (row.get("scenario", ""), row.get("controller", ""), row.get("scope", ""), seed)

    rows.sort(key=sort_key)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("Merged %d files -> %s (%d rows)", len(paths), out_path, len(rows))
    return len(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Merge sharded metric CSVs.")
    parser.add_argument("--pattern", required=True, help='Glob, e.g. "outputs/tmp_marl/*.csv".')
    parser.add_argument("--out", required=True, help="Destination CSV path.")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()
    merge(args.pattern, args.out, dedupe=not args.keep_duplicates)


if __name__ == "__main__":
    main()
