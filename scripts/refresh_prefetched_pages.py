#!/usr/bin/env python3
"""Download citation pages into tests/fixtures/pages for offline test runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from internal.service.validator import HttpxUrlFetcher

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "tests" / "fixtures" / "pages"
MANIFEST_PATH = PAGES_DIR / "manifest.json"

URLS = {
    "https://fastapi.tiangolo.com/": "fastapi_home.html",
    "https://fastapi.tiangolo.com/features/": "fastapi_features.html",
    "https://www.python.org/doc/essays/blurb/": "python_blurb.html",
    "https://docs.python.org/3/library/asyncio.html": "asyncio.html",
}


async def main() -> None:
    """Fetch each manifest URL and rewrite the local HTML snapshot files."""

    fetcher = HttpxUrlFetcher(timeout_seconds=20.0)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for url, filename in URLS.items():
        body = await fetcher.fetch(url)
        (PAGES_DIR / filename).write_text(body, encoding="utf-8")
        manifest[url] = filename
        print(f"wrote {filename} ({len(body)} bytes)")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
