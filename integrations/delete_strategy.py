"""
Delete strategy recommendations
===============================

Evaluation (native synthetic + hnswlib) shows:

* Soft-delete wins pure recall but leaves residual floats (privacy fail).
* MN-RU heal can collapse recall under multi-wave deletes — **experimental**.
* Wipe + compact/rebuild keeps residual-safe usable search — **recommended**.

Production default: physical wipe + ANN rebuild (compact). Heal is optional
only for tiny fractions when explicitly measured safe.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

StrategyAction = Literal[
    "wipe_only",
    "wipe_and_heal",
    "wipe_and_compact",
    "wipe_heal_and_compact",
    "full_rebuild_recommended",
]


@dataclass
class StrategyRecommendation:
    action: StrategyAction
    reason: str
    delete_fraction: float
    prefer_compact: bool
    prefer_heal: bool
    prefer_full_rebuild: bool
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def heal_allowed_by_env() -> bool:
    """MN-RU heal is opt-in (experimental). Default off for product paths."""
    return os.environ.get("HEALER_ALLOW_HEAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def recommend_delete_strategy(
    *,
    delete_count: int,
    index_size: int,
    backend_supports_compact: bool = True,
    backend_supports_heal: bool = True,
    low_frac: float = 0.01,
    high_frac: float = 0.05,
    critical_frac: float = 0.10,
    allow_heal: bool | None = None,
) -> StrategyRecommendation:
    """
    Recommend unlearning action given batch size and index size.

    Defaults bias strongly toward **wipe + compact/rebuild**. Heal is only
    suggested when ``allow_heal`` is True (or ``HEALER_ALLOW_HEAL=1``) and
    the fraction is below ``low_frac``.
    """
    if allow_heal is None:
        allow_heal = heal_allowed_by_env()

    if index_size <= 0:
        frac = 1.0 if delete_count > 0 else 0.0
    else:
        frac = float(delete_count) / float(index_size)

    if delete_count <= 0:
        return StrategyRecommendation(
            action="wipe_only",
            reason="empty_batch",
            delete_fraction=frac,
            prefer_compact=False,
            prefer_heal=False,
            prefer_full_rebuild=False,
            details="no deletes",
        )

    if not backend_supports_compact:
        # No compact: heal only if explicitly allowed; else wipe-only + warn.
        if allow_heal and backend_supports_heal:
            return StrategyRecommendation(
                action="wipe_and_heal",
                reason="no_compact_backend_heal_opt_in",
                delete_fraction=frac,
                prefer_compact=False,
                prefer_heal=True,
                prefer_full_rebuild=False,
                details="no compact(); heal is experimental — measure recall",
            )
        return StrategyRecommendation(
            action="wipe_only",
            reason="no_compact_backend",
            delete_fraction=frac,
            prefer_compact=False,
            prefer_heal=False,
            prefer_full_rebuild=False,
            details="wipe only; schedule offline rebuild when possible",
        )

    if frac >= critical_frac:
        return StrategyRecommendation(
            action="full_rebuild_recommended",
            reason="critical_delete_fraction",
            delete_fraction=frac,
            prefer_compact=True,
            prefer_heal=False,
            prefer_full_rebuild=True,
            details=(
                f"delete_fraction={frac:.3f} >= {critical_frac}: "
                "wipe + full compact/rebuild (do not rely on MN-RU)"
            ),
        )

    if frac >= high_frac:
        return StrategyRecommendation(
            action="wipe_and_compact",
            reason="elevated_delete_fraction",
            delete_fraction=frac,
            prefer_compact=True,
            prefer_heal=False,
            prefer_full_rebuild=True,
            details=(
                f"delete_fraction={frac:.3f} >= {high_frac}: "
                "wipe + always compact (recommended product path)"
            ),
        )

    # Small batch: still prefer compact; heal only if opt-in
    if allow_heal and backend_supports_heal and frac < low_frac:
        return StrategyRecommendation(
            action="wipe_heal_and_compact",
            reason="tiny_batch_heal_opt_in",
            delete_fraction=frac,
            prefer_compact=True,
            prefer_heal=True,
            prefer_full_rebuild=False,
            details=(
                "tiny batch: wipe + experimental heal + compact; "
                "set HEALER_ALLOW_HEAL=0 to disable heal"
            ),
        )

    return StrategyRecommendation(
        action="wipe_and_compact",
        reason="default_product_path",
        delete_fraction=frac,
        prefer_compact=True,
        prefer_heal=False,
        prefer_full_rebuild=False,
        details=(
            "default: wipe + compact (residual-safe usable search). "
            "MN-RU heal is experimental (HEALER_ALLOW_HEAL=1)."
        ),
    )


def compact_mode_for_recommendation(
    rec: StrategyRecommendation,
    *,
    coalesce_ok: bool = True,
) -> str:
    """
    Map recommendation to ErasureService ``compact`` argument.

    Returns ``always``, ``auto``, or ``never``.
    """
    if rec.prefer_full_rebuild:
        return "always"
    if rec.prefer_compact:
        # Prefer always for residual-free structure unless coalesce explicitly OK
        # and fraction is small.
        if rec.delete_fraction >= 0.05 or not coalesce_ok:
            return "always"
        return "auto"
    return "never"


def apply_strategy_to_delete_kwargs(
    *,
    delete_count: int,
    index_size: int,
    backend_supports_compact: bool,
    backend_supports_heal: bool = False,
    compact: str | bool | None = None,
) -> dict[str, Any]:
    """
    When compact is None/auto and adaptive strategy is enabled, force compact
    mode from the recommendation.
    """
    adaptive = os.environ.get("HEALER_ADAPTIVE_COMPACT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
        "",
    )
    rec = recommend_delete_strategy(
        delete_count=delete_count,
        index_size=index_size,
        backend_supports_compact=backend_supports_compact,
        backend_supports_heal=backend_supports_heal,
    )
    out: dict[str, Any] = {"strategy": rec.to_dict()}
    if compact is not None and compact not in ("auto", True):
        out["compact"] = compact
        return out
    if not adaptive:
        out["compact"] = compact if compact is not None else "auto"
        return out
    out["compact"] = compact_mode_for_recommendation(rec)
    return out
