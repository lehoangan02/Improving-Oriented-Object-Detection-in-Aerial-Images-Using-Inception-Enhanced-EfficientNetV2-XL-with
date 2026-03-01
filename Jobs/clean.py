"""Remove wget-style progress lines from download_dota_train.log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROGRESS_LINE = re.compile(
    r"^\s*\d+K(?:\s+\.+)+\s+\d+%\s+[0-9.]+[KMGTP]?B?\s+\S+\s*$"
)


def clean_log(log_path: Path) -> tuple[int, int]:
    """Filter the given log file in-place and return (removed, total) counts."""

    lines = log_path.read_text().splitlines(keepends=True)
    kept: list[str] = []
    removed = 0

    for line in lines:
        if PROGRESS_LINE.match(line.rstrip("\r\n")):
            removed += 1
            continue
        kept.append(line)

    log_path.write_text("".join(kept))
    return removed, len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logfile",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("download_dota_train.log"),
        help="Path to the download log file to clean",
    )
    args = parser.parse_args()

    removed, total = clean_log(args.logfile)
    print(f"Removed {removed} / {total} lines from {args.logfile}")


if __name__ == "__main__":
    main()
