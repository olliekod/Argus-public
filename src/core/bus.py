# Created by Oliver Meihls

# Argus Event Bus
#
# Central Pub/Sub event bus with dedicated worker threads per topic.
#
# Design constraints
# * ``publish()`` is **O(1)** — it only appends to a ``collections.deque``
# and **never** calls handlers directly.
# * Each topic has its own worker thread that drains the queue and calls
# registered handlers sequentially.
# * Back-pressure uses ``deque(maxlen=…)`` with DROP_OLD semantics for
# ``market.*`` topics (newest data wins).

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("argus.bus")

# Default queue capacities per topic prefix
_DEFAULT_MAXLEN: Dict[str, int] = {
    "market.": 50_000,     # market data is high-volume, drop old
    "signals.": 10_000,    # signals should not be lost easily
    "system.": 5_000,      # heartbeats, status
}

# Sentinel object to signal worker shutdown
_STOP = object()


def _maxlen_for(topic: str) -> int:
    # Choose a deque maxlen based on topic prefix.
    for prefix, maxlen in _DEFAULT_MAXLEN.items():
        if topic.startswith(prefix):
            return maxlen
    return 10_000


class EventBus:
    # Thread-safe Pub/Sub event bus with async worker drains.
    #
    # Usage::
    #
    # bus = EventBus()
    # bus.subscribe("market.quotes", my_handler)
    # bus.start()
    # bus.publish("market.quotes", quote_event)
    # ...
    # bus.stop()                # graceful shutdown

    # After this many consecutive errors a handler is automatically
    # disabled (circuit-broken) to prevent unbounded error-log growth.
    _CIRCUIT_BREAKER_THRESHOLD: int = 100

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queues: Dict[str, collections.deque] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._events: Dict[str, threading.Event] = {}
        self._running = False
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, int]] = {}
        self._queue_peak: Dict[str, int] = {}

        # Circuit-breaker state — keyed by handler id(…) for O(1) lookup
        self._handler_consecutive_errors: Dict[int, int] = {}
        self._circuit_broken_handlers: set = set()

    # ──── public API ────────────────────────────────────────

    def subscribe(self, topic: str, handler: Callable) -> None:
        # Register *handler* for *topic*.
        #
        # Must be called **before** :meth:`start`.
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
                self._queues[topic] = collections.deque(
                    maxlen=_maxlen_for(topic)
                )
                self._events[topic] = threading.Event()
                self._stats[topic] = {
                    "published": 0,
                    "processed": 0,
                    "dropped": 0,
                    "errors": 0,
                    "last_worker_lag_ms": None,
                    "avg_worker_lag_ms": None,
                }
                self._queue_peak[topic] = 0
            self._subscribers[topic].append(handler)
        logger.debug("subscribed %s to %s", handler.__qualname__, topic)

    def publish(self, topic: str, event: Any) -> None:
        # Enqueue *event* on *topic*.
        #
        # O(1) — never blocks, never calls handlers.
        # If the deque is full the oldest item is silently dropped
        # (DROP_OLD back-pressure for ``market.*``).
        q = self._queues.get(topic)
        if q is None:
            return  # no subscribers for this topic
        was_full = len(q) == q.maxlen
        q.append(event)
        depth = len(q)
        if depth > self._queue_peak.get(topic, 0):
            self._queue_peak[topic] = depth
        if was_full:
            self._stats[topic]["dropped"] += 1
        self._stats[topic]["published"] += 1
        # Wake the worker
        ev = self._events.get(topic)
        if ev is not None:
            ev.set()

    def start(self) -> None:
        # Spawn a worker thread for every registered topic.
        if self._running:
            return
        self._running = True
        with self._lock:
            for topic in list(self._subscribers):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(topic,),
                    name=f"bus-{topic}",
                    daemon=True,
                )
                self._workers[topic] = t
                t.start()
        logger.info(
            "EventBus started — %d topic(s): %s",
            len(self._workers),
            ", ".join(sorted(self._workers)),
        )

    def stop(self, timeout: float = 5.0) -> None:
        # Signal all workers to drain and exit.
        if not self._running:
            return
        self._running = False
        # Push a sentinel into every queue so workers wake up
        for topic, q in self._queues.items():
            q.append(_STOP)
            ev = self._events.get(topic)
            if ev:
                ev.set()
        # Join all workers
        for topic, t in self._workers.items():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("bus worker %s did not exit in time", topic)
        self._workers.clear()
        logger.info("EventBus stopped")

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        # Return per-topic publish / processed / dropped counters.
        return dict(self._stats)

    def get_queue_depths(self) -> Dict[str, int]:
        # Current number of events waiting in each topic queue.
        return {topic: len(q) for topic, q in self._queues.items()}

    def get_status_summary(self) -> Dict[str, Any]:
        # Return enriched per-topic status metrics for /status.
        summary: Dict[str, Any] = {}
        for topic, stats in self._stats.items():
            summary[topic] = {
                "events_published": stats.get("published", 0),
                "handler_calls_processed": stats.get("processed", 0),
                "dropped_events": stats.get("dropped", 0),
                "worker_errors": stats.get("errors", 0),
                "queue_depth": len(self._queues.get(topic, [])),
                "peak_depth": self._queue_peak.get(topic, 0),
                "last_worker_lag_ms": stats.get("last_worker_lag_ms"),
                "avg_worker_lag_ms": stats.get("avg_worker_lag_ms"),
            }
        return summary

    def is_running(self) -> bool:
        # Return True if the bus workers are running.
        return self._running

    # ──── internal ──────────────────────────────────────────

    def _worker_loop(self, topic: str) -> None:
        # Drain the queue for *topic*, calling handlers for each event.
        q = self._queues[topic]
        ev = self._events[topic]
        handlers = self._subscribers.get(topic, [])
        stats = self._stats[topic]

        logger.debug("worker started for %s (%d handlers)", topic, len(handlers))

        while True:
            # Block until there's something to process or we're told to stop
            ev.wait(timeout=1.0)
            ev.clear()

            # Drain in batches
            while q:
                try:
                    item = q.popleft()
                except IndexError:
                    break

                if item is _STOP:
                    # Drain remaining real events before exiting
                    if not self._running:
                        self._drain_remaining(topic, q, handlers, stats)
                        logger.debug("worker %s exiting", topic)
                        return
                    continue

                if hasattr(item, "receive_time"):
                    lag_ms = (time.time() - getattr(item, "receive_time")) * 1000
                    stats["last_worker_lag_ms"] = round(lag_ms, 2)
                    prev = stats.get("avg_worker_lag_ms")
                    if prev is None:
                        stats["avg_worker_lag_ms"] = round(lag_ms, 2)
                    else:
                        stats["avg_worker_lag_ms"] = round((lag_ms * 0.2) + (prev * 0.8), 2)

                for handler in handlers:
                    hid = id(handler)
                    if hid in self._circuit_broken_handlers:
                        continue
                    try:
                        handler(item)
                        stats["processed"] += 1
                        # Reset on success
                        self._handler_consecutive_errors[hid] = 0
                    except Exception:
                        stats["errors"] += 1
                        consec = self._handler_consecutive_errors.get(hid, 0) + 1
                        self._handler_consecutive_errors[hid] = consec
                        if consec >= self._CIRCUIT_BREAKER_THRESHOLD:
                            self._circuit_broken_handlers.add(hid)
                            logger.error(
                                "CIRCUIT BREAKER: handler %s disabled after "
                                "%d consecutive errors on topic %s",
                                handler.__qualname__,
                                consec,
                                topic,
                            )
                        else:
                            logger.exception(
                                "handler %s raised on topic %s",
                                handler.__qualname__,
                                topic,
                            )

            if not self._running:
                return

    def _drain_remaining(
        self,
        topic: str,
        q: collections.deque,
        handlers: List[Callable],
        stats: Dict[str, int],
    ) -> None:
        # Process any events still in the queue before shutdown.
        while q:
            try:
                item = q.popleft()
            except IndexError:
                break
            if item is _STOP:
                continue
            for handler in handlers:
                hid = id(handler)
                if hid in self._circuit_broken_handlers:
                    continue
                try:
                    handler(item)
                    stats["processed"] += 1
                    # Reset on success
                    self._handler_consecutive_errors[hid] = 0
                except Exception:
                    stats["errors"] += 1
                    consec = self._handler_consecutive_errors.get(hid, 0) + 1
                    self._handler_consecutive_errors[hid] = consec
                    if consec >= self._CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_broken_handlers.add(hid)
                        logger.error(
                            "CIRCUIT BREAKER: handler %s disabled after "
                            "%d consecutive errors during drain on %s",
                            handler.__qualname__,
                            consec,
                            topic,
                        )
                    else:
                        logger.exception(
                            "handler %s raised during drain on %s",
                            handler.__qualname__,
                            topic,
                        )
