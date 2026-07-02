"""Load prefetched citation pages from disk for deterministic tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
PAGES_DIR = FIXTURES_DIR / "pages"
MANIFEST_PATH = PAGES_DIR / "manifest.json"


def _load_url_map(pages_dir: Path, manifest_path: Path) -> dict[str, Path]:
    """Read the URL manifest and resolve each entry to an on-disk page path."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {url: pages_dir / filename for url, filename in manifest.items()}


@dataclass
class PrefetchedPageFetcher:
    """Serve citation URL content from checked-in HTML snapshots.

    Sample payloads keep real documentation URLs. Tests read matching HTML from
    ``tests/fixtures/pages`` so eval runs stay offline and deterministic.
    """

    pages_dir: Path = PAGES_DIR
    manifest_path: Path = MANIFEST_PATH
    url_to_file: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.url_to_file:
            self.url_to_file = _load_url_map(self.pages_dir, self.manifest_path)

    async def fetch(self, url: str) -> str:
        """Return prefetched HTML for ``url`` or raise like an unreachable source."""

        page_path = self.url_to_file.get(url)
        if page_path is None or not page_path.is_file():
            raise FileNotFoundError(f"prefetched page not found for url: {url}")
        return page_path.read_text(encoding="utf-8")


def prefetched_fetcher() -> PrefetchedPageFetcher:
    """Construct the default fetcher wired into unit and evaluation tests."""

    return PrefetchedPageFetcher()
