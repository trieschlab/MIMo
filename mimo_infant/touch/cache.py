from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterator


class StatsCache(MutableMapping):
    """Small statistics wrapper around a cachetools cache.
    """

    def __init__(self, cache: MutableMapping):
        self._cache = cache
        self._hits = 0
        self._misses = 0

    def __getitem__(self, key: Any) -> Any:
        try:
            value = self._cache[key]
        except KeyError:
            self._misses += 1
            raise
        else:
            self._hits += 1
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._cache[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._cache[key]

    def __iter__(self) -> Iterator:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def hits(self) -> int:
        return self._hits

    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def currsize(self):
        return getattr(self._cache, "currsize", len(self._cache))

    @property
    def maxsize(self):
        return getattr(self._cache, "maxsize", None)