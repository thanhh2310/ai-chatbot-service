import time
from threading import RLock
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")


class TTLCache:
    def __init__(self, ttl_seconds: float, max_size: int = 512):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._items: dict[Hashable, tuple[float, object]] = {}
        self._lock = RLock()

    def get(self, key: Hashable):
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            return value

    def set(self, key: Hashable, value):
        now = time.monotonic()
        with self._lock:
            if len(self._items) >= self.max_size:
                oldest_key = min(self._items, key=lambda k: self._items[k][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (now + self.ttl_seconds, value)

    def get_or_set(self, key: Hashable, factory: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value
