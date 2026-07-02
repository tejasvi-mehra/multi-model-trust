"""Bounded async execution helpers for concurrent I/O workloads."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def map_bounded(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    concurrency: int,
) -> list[R]:
    """Run async work with a concurrency limit and preserve input ordering."""

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return await asyncio.gather(*(guarded(item) for item in items))
