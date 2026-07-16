"""
In-process observability metrics for HNSW Healer.

Exposes counters and timing histograms suitable for ``GET /metrics``
(Prometheus text format) and structured JSON via ``snapshot()``.

Not a full Prometheus client — zero extra dependencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _TimerAgg:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def observe(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if ms > self.max_ms:
            self.max_ms = ms

    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


class MetricsRegistry:
    """Thread-safe process metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.timers: dict[str, _TimerAgg] = {}
        self.gauges: dict[str, float] = {}
        self.started_unix = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = float(value)

    def observe_ms(self, name: str, ms: float) -> None:
        with self._lock:
            if name not in self.timers:
                self.timers[name] = _TimerAgg()
            self.timers[name].observe(float(ms))

    def time_block(self, name: str):
        """Context manager that records elapsed milliseconds."""
        registry = self

        class _Block:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                ms = (time.perf_counter() - self_inner.t0) * 1000.0
                registry.observe_ms(name, ms)
                return False

        return _Block()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": time.time() - self.started_unix,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers_ms": {
                    k: {
                        "count": v.count,
                        "total_ms": v.total_ms,
                        "mean_ms": v.mean_ms(),
                        "max_ms": v.max_ms,
                    }
                    for k, v in self.timers.items()
                },
            }

    def prometheus_text(self) -> str:
        """Minimal Prometheus exposition format."""
        lines: list[str] = [
            "# HELP hnsw_healer_info HNSW Healer process metrics",
            "# TYPE hnsw_healer_info gauge",
            f"hnsw_healer_info{{version=\"0.3.0\"}} 1",
        ]
        snap = self.snapshot()
        lines.append(
            f"hnsw_healer_uptime_seconds {snap['uptime_s']:.3f}"
        )
        for name, val in sorted(snap["counters"].items()):
            prom = name.replace(".", "_").replace("-", "_")
            lines.append(f"hnsw_healer_{prom}_total {val}")
        for name, val in sorted(snap["gauges"].items()):
            prom = name.replace(".", "_").replace("-", "_")
            lines.append(f"hnsw_healer_{prom} {val}")
        for name, stats in sorted(snap["timers_ms"].items()):
            prom = name.replace(".", "_").replace("-", "_")
            lines.append(
                f"hnsw_healer_{prom}_count {stats['count']}"
            )
            lines.append(
                f"hnsw_healer_{prom}_sum_ms {stats['total_ms']:.6f}"
            )
            lines.append(
                f"hnsw_healer_{prom}_max_ms {stats['max_ms']:.6f}"
            )
        return "\n".join(lines) + "\n"


# Process-global default registry
METRICS = MetricsRegistry()
