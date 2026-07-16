"""Guard against documentation / example fail-open footguns."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_golden_path_does_not_recommend_manual_compact_only() -> None:
    text = _read("docs/GOLDEN_PATH.md")
    assert "no manual" in text.lower() or "automatically" in text.lower() or "auto" in text.lower()
    # Should not show success after compact=never without warning context
    if "compact=never" in text or 'compact="never"' in text:
        assert "warn" in text.lower() or "risk" in text.lower()


def test_examples_do_not_set_fail_closed_false() -> None:
    for path in (ROOT / "examples").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "fail_closed=False" not in src, f"{path} disables fail_closed"


def test_chroma_example_asserts_receipt_success() -> None:
    src = _read("examples/chroma_forget/run.py")
    assert "receipt.success" in src
    assert "compacted" in src


def test_attack_demo_exists() -> None:
    assert (ROOT / "examples" / "attack_demo" / "run.py").is_file()
