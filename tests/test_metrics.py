"""Metrics registry tests."""

from __future__ import annotations

from api.metrics import MetricsRegistry


def test_metrics_inc_and_prometheus() -> None:
    m = MetricsRegistry()
    m.inc("deletes_batches", 2)
    m.observe_ms("heal", 12.5)
    m.set_gauge("compact_pending", 3)
    snap = m.snapshot()
    assert snap["counters"]["deletes_batches"] == 2
    assert snap["timers_ms"]["heal"]["count"] == 1
    text = m.prometheus_text()
    assert "hnsw_healer_deletes_batches_total" in text
    assert "hnsw_healer_heal_count" in text
