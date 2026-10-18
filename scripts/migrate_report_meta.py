#!/usr/bin/env python
"""Upgrades every old `meta.json` under a folder to the current `ReportMeta` structure, in place.

    uv run python scripts/migrate_report_meta.py reports

- A file already in the current structure is left untouched.
- A file that matches neither structure is listed and skipped.
"""

import argparse
from pathlib import Path

from pydantic import ValidationError

from in_tfm.report_meta import META_NAME, ReportMeta, parses_as_legacy


def migrate_all(root: Path) -> None:
    migrated, current, unreadable = 0, 0, []
    for meta_path in sorted(root.rglob(META_NAME)):
        was_legacy = parses_as_legacy(meta_path.read_text())
        try:
            ReportMeta.migrate_file(meta_path)
        except ValidationError:
            unreadable.append(meta_path)
            continue
        if was_legacy:
            migrated += 1
        else:
            current += 1
    print(f"migrated {migrated}, already current {current}, unreadable {len(unreadable)}")
    for meta_path in unreadable:
        print(f"  unreadable: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    migrate_all(parser.parse_args().root)


if __name__ == "__main__":
    main()
