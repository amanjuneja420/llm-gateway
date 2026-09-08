"""
In-memory token-bucket rate limiter, per client key.
========================================================
Deliberately simple, matching the rest of this project: in-process state
only (a plain dict), no Redis or other external store, so it does not
survive a restart and does not work across multiple gateway instances -
fine for an MVP that only ever runs as one process.

Each client key gets its own bucket. The key comes from a request header
(X-API-Key by default - see RATE_LIMIT_HEADER in main.py) that is NOT
validated against any real authentication system; this is rate limiting,
not auth. A request with no header at all is bucketed under "anonymous"
rather than rejected or exempted.

Token bucket, not a fixed window counter: a bucket starts full (capacity
tokens) and refills continuously at capacity/window_seconds tokens per
second, capped at capacity. This allows a legitimate burst up to the full
per-window limit immediately, then smooths out to the steady-state rate -
the standard behavior for "N requests per minute" APIs, as opposed to a
fixed window that would let a client do N requests in the last second of
one window and another N in the first second of the next.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity  # start full - an immediate burst up to capacity is allowed
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_consume(self, amount: float = 1.0) -> tuple[bool, float]:
        """
        Try to consume `amount` tokens. Returns (allowed, retry_after_seconds).
        retry_after_seconds is 0.0 when allowed; otherwise it's how long
        until enough tokens will have refilled for this same request to
        succeed - suitable for a Retry-After header.
        """
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        deficit = amount - self.tokens
        retry_after = deficit / self.refill_rate
        return False, retry_after


class RateLimiter:
    """One TokenBucket per client key, created lazily on first use."""

    def __init__(self, capacity: float, window_seconds: float):
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.refill_rate = capacity / window_seconds
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, client_key: str) -> tuple[bool, float]:
        bucket = self._buckets.get(client_key)
        if bucket is None:
            bucket = TokenBucket(capacity=self.capacity, refill_rate=self.refill_rate)
            self._buckets[client_key] = bucket
        return bucket.try_consume()
