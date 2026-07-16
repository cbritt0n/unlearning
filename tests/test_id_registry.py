"""Tests for collection-scoped ID registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from integrations.id_registry import CollectionIdRegistry, IdMappingError


def test_register_and_resolve() -> None:
    reg = CollectionIdRegistry()
    a = reg.register("users", "alice")
    b = reg.register("users", "bob")
    assert a != b
    assert reg.resolve("users", "alice").label == a
    assert reg.external_of("users", b) == "bob"


def test_collections_are_isolated() -> None:
    reg = CollectionIdRegistry()
    u = reg.register("users", "x")
    d = reg.register("docs", "x")
    # Same external string may share label value across collections.
    assert reg.resolve("users", "x").collection == "users"
    assert reg.resolve("docs", "x").collection == "docs"
    assert u == 0 and d == 0


def test_drop_and_unknown() -> None:
    reg = CollectionIdRegistry()
    reg.register("c", "id1")
    reg.drop("c", "id1")
    with pytest.raises(IdMappingError):
        reg.resolve("c", "id1")


def test_persist_roundtrip(tmp_path: Path) -> None:
    reg = CollectionIdRegistry()
    reg.register("c", "a")
    reg.register("c", "b")
    path = tmp_path / "ids.json"
    reg.save(path)

    reg2 = CollectionIdRegistry()
    reg2.load(path)
    assert reg2.resolve("c", "b").label == reg.resolve("c", "b").label
