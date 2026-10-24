"""Checks that every link and image in a folder of report pages lands, anchors included.

- The reports promise that every cluster named is a link to that cluster, so a run checks it rather than trusting it.
- Only local references are checked. A page's own `#anchor` must be an `id` on that page.
- An anchor of `key=value` pairs is state for the page's script, such as a filter, so only its file is checked.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

REFERENCE = re.compile(r'(?:src|href)="([^"]+)"')
ELEMENT_ID = re.compile(r'\bid="([^"]+)"')
EXTERNAL = ("http://", "https://", "data:", "mailto:")


class BrokenLink(BaseModel):
    page: Path
    reference: str


class LinkCheck(BaseModel):
    pages: int
    checked: int
    broken: list[BrokenLink]


def local_references(html: str) -> Iterator[str]:
    return (ref for ref in REFERENCE.findall(html) if not ref.startswith(EXTERNAL))


def lands(page: Path, reference: str, ids_of_page: dict[Path, set[str]]) -> bool:
    path, _, anchor = reference.partition("#")
    target = (page.parent / path).resolve() if path else page.resolve()
    if not target.exists():
        return False
    if not anchor or "=" in anchor:
        return True
    if target not in ids_of_page:
        ids_of_page[target] = set(ELEMENT_ID.findall(target.read_text()))
    return anchor in ids_of_page[target]


def check_links(root: Path) -> LinkCheck:
    pages = sorted(root.rglob("*.html"))
    ids_of_page: dict[Path, set[str]] = {}
    checked, broken = 0, []
    for page in pages:
        for reference in local_references(page.read_text()):
            checked += 1
            if not lands(page, reference, ids_of_page):
                broken.append(BrokenLink(page=page.relative_to(root), reference=reference))
    return LinkCheck(pages=len(pages), checked=checked, broken=broken)
