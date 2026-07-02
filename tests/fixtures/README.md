# Prefetched citation pages

Tests keep the same citation URLs as the sample payloads but read HTML from this directory.

- `manifest.json` maps each URL to a filename in this folder.
- `PrefetchedPageFetcher` in `prefetched_fetcher.py` loads page bodies during tests.
- URLs missing from the manifest simulate unreachable sources.

Refresh snapshots when sample quotes change:

```bash
python scripts/refresh_prefetched_pages.py
```

If the refresh script is unavailable, fetch the four manifest URLs manually and save them here, then update `manifest.json`.
