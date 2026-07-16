"""Chroma hard-delete hook tests.

Integration tests (real chromadb + hnswlib) skip when optional deps are missing.
Fail-closed matrix tests use mocks and always run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.backends import BackendEraseResult
from integrations.chroma_hook import ChromaHardDeleteCollection
from integrations.erase_service import ErasureService
from integrations.id_registry import CollectionIdRegistry


# ---------------------------------------------------------------------------
# Integration (optional deps)
# ---------------------------------------------------------------------------


def test_chroma_delete_hard_erases_backend() -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("hnswlib")
    import chromadb
    import numpy as np

    from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

    reg = CollectionIdRegistry()
    backend = HnswlibHardDeleteAdapter(
        dim=4,
        max_elements=50,
        registry=reg,
        collection="docs",
        enable_heal_mirror=False,
    )
    svc = ErasureService(reg, backend, drop_mappings=True)

    client = chromadb.Client()
    raw = client.get_or_create_collection("docs")
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )

    emb = np.random.randn(3, 4).astype(np.float32)
    ids = ["d0", "d1", "d2"]
    wrapped.add(ids=ids, embeddings=emb.tolist(), documents=["a", "b", "c"])

    assert reg.contains("docs", "d1")
    label = reg.resolve("docs", "d1").label
    assert not np.all(backend.get_vector(label) == 0.0)

    receipt = wrapped.delete(ids=["d1"], reason="test")
    assert receipt is not None
    assert receipt.success
    assert receipt.compacted is True
    assert receipt.residual_proof is not None
    assert backend.verify_zeroed("docs", label)

    got = raw.get(ids=["d1"])
    assert got["ids"] == [] or "d1" not in got["ids"]


def test_chroma_register_on_add_no_manual_register() -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("hnswlib")
    import chromadb

    from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter

    reg = CollectionIdRegistry()
    backend = HnswlibHardDeleteAdapter(
        dim=4,
        max_elements=20,
        registry=reg,
        collection="docs",
        enable_heal_mirror=False,
    )
    svc = ErasureService(reg, backend)
    client = chromadb.Client()
    raw = client.get_or_create_collection("docs_reg")
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )
    emb = [[0.1, 0.2, 0.3, 0.4]]
    wrapped.add(ids=["only"], embeddings=emb, documents=["x"])
    assert reg.contains("docs", "only")


# ---------------------------------------------------------------------------
# Fail-closed matrix (ticket 5) — no chromadb/hnswlib required
# ---------------------------------------------------------------------------


class _FailEraseBackend:
    def hard_delete_label(self, collection, label, *, max_m=16):
        return BackendEraseResult(
            success=False, label=label, message="simulated erase failure"
        )

    def verify_zeroed(self, collection, label) -> bool:
        return False

    def get_vector(self, label):
        return [1.0, 1.0]


class _FailCompactBackend:
    def __init__(self) -> None:
        self._zeroed: set[int] = set()
        self._vecs = {0: [1.0, 2.0, 3.0, 4.0], 1: [0.5, 0.5, 0.5, 0.5]}

    def hard_delete_label(self, collection, label, *, max_m=16):
        self._zeroed.add(label)
        self._vecs[label] = [0.0, 0.0, 0.0, 0.0]
        return BackendEraseResult(
            success=True, label=label, bytes_wiped=16, message="zeroed"
        )

    def verify_zeroed(self, collection, label) -> bool:
        return label in self._zeroed

    def get_vector(self, label):
        return list(self._vecs[label])

    def compact(self) -> int:
        raise RuntimeError("simulated compact failure")


class _FailProofBackend:
    def hard_delete_label(self, collection, label, *, max_m=16):
        return BackendEraseResult(
            success=True, label=label, bytes_wiped=16, message="zeroed"
        )

    def verify_zeroed(self, collection, label) -> bool:
        return True

    def get_vector(self, label):
        return [3.0, 3.0, 3.0, 3.0]


def _mock_chroma_collection() -> MagicMock:
    raw = MagicMock()
    raw.name = "docs"
    raw.add = MagicMock()
    raw.delete = MagicMock()
    raw.get = MagicMock(return_value={"ids": []})
    return raw


def test_chroma_fail_closed_on_erase_failure() -> None:
    reg = CollectionIdRegistry()
    reg.register("docs", "d0", label=0)
    svc = ErasureService(reg, _FailEraseBackend(), drop_mappings=False)
    raw = _mock_chroma_collection()
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )

    with pytest.raises(RuntimeError, match="hard erase failed"):
        wrapped.delete(ids=["d0"], residual_proof="off")

    raw.delete.assert_not_called()


def test_chroma_fail_closed_on_compact_failure() -> None:
    reg = CollectionIdRegistry()
    reg.register("docs", "d0", label=0)
    svc = ErasureService(
        reg, _FailCompactBackend(), drop_mappings=False, default_compact="always"
    )
    raw = _mock_chroma_collection()
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )

    with pytest.raises(RuntimeError, match="hard erase failed"):
        wrapped.delete(ids=["d0"], residual_proof="off", compact="always")

    raw.delete.assert_not_called()


def test_chroma_fail_closed_on_residual_proof_failure() -> None:
    reg = CollectionIdRegistry()
    reg.register("docs", "d0", label=0)
    svc = ErasureService(reg, _FailProofBackend(), drop_mappings=False)
    raw = _mock_chroma_collection()
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )

    with pytest.raises(RuntimeError, match="hard erase failed"):
        wrapped.delete(ids=["d0"], residual_proof="full", compact="never")

    raw.delete.assert_not_called()


def test_chroma_fail_open_allows_metadata_delete() -> None:
    reg = CollectionIdRegistry()
    reg.register("docs", "d0", label=0)
    svc = ErasureService(reg, _FailEraseBackend(), drop_mappings=False)
    raw = _mock_chroma_collection()
    wrapped = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=False
    )
    receipt = wrapped.delete(ids=["d0"], residual_proof="off")
    assert receipt is not None
    assert receipt.success is False
    raw.delete.assert_called_once()
