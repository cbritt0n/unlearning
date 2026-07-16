#!/usr/bin/env python3
"""
Golden path: Chroma + hnswlib hard-delete with auto compact + residual proof.

Usage (from repo root)::

    pip install -e ".[chroma]"
    python examples/chroma_forget/run.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

try:
    import chromadb
except ImportError:
    print("Install chromadb: pip install -e \".[chroma]\"", file=sys.stderr)
    sys.exit(1)

try:
    import hnswlib  # noqa: F401
except ImportError:
    print("Install hnswlib: pip install -e \".[chroma]\"", file=sys.stderr)
    sys.exit(1)

from integrations import CollectionIdRegistry, ErasureService
from integrations.chroma_hook import ChromaHardDeleteCollection
from integrations.hnswlib_adapter import HnswlibHardDeleteAdapter


def main() -> int:
    dim = 8
    reg = CollectionIdRegistry()
    backend = HnswlibHardDeleteAdapter(
        dim=dim,
        max_elements=100,
        registry=reg,
        collection="docs",
        enable_heal_mirror=False,
    )
    svc = ErasureService(reg, backend, drop_mappings=True)

    client = chromadb.Client()
    raw = client.get_or_create_collection("docs")
    col = ChromaHardDeleteCollection(
        raw, svc, collection_name="docs", fail_closed=True
    )

    rng = np.random.default_rng(0)
    ids = ["alice_doc", "bob_doc", "carol_doc"]
    embeddings = rng.standard_normal((3, dim), dtype=np.float32)
    documents = [
        "Alice private notes",
        "Bob project plan",
        "Carol public FAQ",
    ]

    # Register-on-add: no POST /v1/ids/register needed.
    col.add(ids=ids, embeddings=embeddings.tolist(), documents=documents)
    print("ingested:", ids)

    # Search via Chroma (metadata path).
    hits = col.query(query_embeddings=[embeddings[0].tolist()], n_results=2)
    print("search ids:", hits.get("ids"))

    # Forget Alice — hard wipe + one compact + residual proof + Chroma delete.
    receipt = col.delete(
        ids=["alice_doc"],
        reason="gdpr_art_17",
        request_id="demo-ticket-1",
    )
    assert receipt is not None
    print("\n--- ErasureReceipt v2 ---")
    print(json.dumps(receipt.to_dict(), indent=2))

    assert receipt.success, receipt.errors
    assert receipt.status == "complete"
    assert receipt.compacted
    assert receipt.residual_proof and receipt.residual_proof.get("passed")

    gone = raw.get(ids=["alice_doc"])
    assert gone["ids"] == [] or "alice_doc" not in gone["ids"]
    print("\nChroma metadata: alice_doc removed")
    print("golden path OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
