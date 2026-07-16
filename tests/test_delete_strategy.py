"""Delete strategy recommendation tests (product defaults)."""

from __future__ import annotations

from integrations.delete_strategy import (
    apply_strategy_to_delete_kwargs,
    compact_mode_for_recommendation,
    recommend_delete_strategy,
)


def test_default_prefers_compact_not_heal(monkeypatch) -> None:
    monkeypatch.delenv("HEALER_ALLOW_HEAL", raising=False)
    rec = recommend_delete_strategy(
        delete_count=2, index_size=1000, backend_supports_compact=True
    )
    assert rec.prefer_compact
    assert rec.prefer_heal is False
    assert rec.action == "wipe_and_compact"


def test_critical_fraction_full_rebuild(monkeypatch) -> None:
    monkeypatch.delenv("HEALER_ALLOW_HEAL", raising=False)
    rec = recommend_delete_strategy(
        delete_count=300, index_size=1000, backend_supports_compact=True
    )
    assert rec.prefer_full_rebuild
    assert rec.action == "full_rebuild_recommended"
    assert compact_mode_for_recommendation(rec) == "always"


def test_heal_only_when_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ALLOW_HEAL", "1")
    rec = recommend_delete_strategy(
        delete_count=1,
        index_size=10_000,
        backend_supports_compact=True,
        backend_supports_heal=True,
        allow_heal=True,
    )
    assert rec.prefer_heal is True
    assert "heal" in rec.action


def test_adaptive_kwargs_force_always_on_large_batch(monkeypatch) -> None:
    monkeypatch.setenv("HEALER_ADAPTIVE_COMPACT", "1")
    monkeypatch.delenv("HEALER_ALLOW_HEAL", raising=False)
    out = apply_strategy_to_delete_kwargs(
        delete_count=500,
        index_size=1000,
        backend_supports_compact=True,
        compact=None,
    )
    assert out["compact"] == "always"
