# Created by Oliver Meihls

# Minimal asyncio pub/sub bus adapter for Argus.
#
# Argus uses an internal pub/sub bus where each *topic* is backed by an
# ``asyncio.Queue``.  This module provides the thin ``Bus`` interface that
# the rest of the Kalshi module depends on.
#
# Design notes
# * ``subscribe(topic)`` returns a **new** Queue each time it is called so
# that multiple consumers can independently read from the same topic.
# * ``publish(topic, message)`` fans out to every subscriber Queue for that
# topic without blocking (uses ``put_nowait``; Queues are unbounded by
# default, but callers can set a maxsize if back-pressure is needed).
# * All messages should be dataclass instances from ``models.py``.

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List


class Bus:
    # Async pub/sub hub: topic string → list of asyncio.Queue.

    def __init__(self, *, subscriber_queue_maxsize: int = 10_000) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue[Any]]] = defaultdict(list)
        self._subscriber_queue_maxsize = max(1, subscriber_queue_maxsize)

    async def publish(self, topic: str, message: Any) -> None:
        # Fan-out *message* to every subscriber Queue for *topic*.
        for q in self._subscribers.get(topic, ()):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Back-pressure safety valve: drop oldest and keep the latest tick.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                q.put_nowait(message)

    async def subscribe(self, topic: str) -> asyncio.Queue[Any]:
        # Return a new Queue that will receive future publishes on *topic*.
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._subscriber_queue_maxsize)
        self._subscribers[topic].append(q)
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        # Remove *queue* from *topic* subscribers (best-effort).
        subs = self._subscribers.get(topic)
        if subs:
            try:
                subs.remove(queue)
            except ValueError:
                pass

    def subscriber_count(self, topic: str) -> int:
        # Return how many Queues are listening on *topic*.
        return len(self._subscribers.get(topic, ()))
