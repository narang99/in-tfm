"""Copies a finished report into a folder ready for a web host, with each cluster body in its own file.

    python scripts/host_report.py <report_dir> <hosted_dir>
    aws s3 sync <hosted_dir> s3://<bucket>/<prefix>

- The copy leaves out `hits.pt`, `collected_hits.pt` and `scan.pt`, which only re-render a report.
- See `in_tfm.hosted_report`.
"""

import shutil
import sys
from pathlib import Path

from in_tfm.hosted_report import write_hosted_index

NOT_HOSTED = ("hits.pt", "collected_hits.pt", "scan.pt")


def main() -> None:
    report_dir, hosted_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if hosted_dir.exists():
        raise SystemExit(
            f"{hosted_dir} exists. Pick a new folder, the copy is rewritten in place."
        )
    shutil.copytree(report_dir, hosted_dir, ignore=shutil.ignore_patterns(*NOT_HOSTED))
    index = write_hosted_index(hosted_dir)
    bodies = len(list(hosted_dir.glob("cluster_*/body.html")))
    print(
        f"{index}: {index.stat().st_size / 1e6:.1f} MB, {bodies} cluster bodies split out"
    )


if __name__ == "__main__":
    main()
