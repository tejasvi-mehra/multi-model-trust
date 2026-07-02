import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.prefetched_fetcher import PrefetchedPageFetcher, prefetched_fetcher


@pytest.fixture
def prefetched_page_fetcher() -> PrefetchedPageFetcher:
    """Provide the shared on-disk citation source fetcher for tests."""

    return prefetched_fetcher()
