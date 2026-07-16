"""
Compliance helpers: residual verification, crypto-shred, threat-model tools.

These modules support enterprise unlearning claims; they do not replace a
full DPIA or legal review.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CryptoShredVault",
    "KmsCryptoShredVault",
    "LocalFileKMS",
    "ResidualProof",
    "ShredReceipt",
    "prove_vector_erased",
    "FragmentationBound",
]


def __getattr__(name: str) -> Any:
    if name in ("CryptoShredVault", "ShredReceipt"):
        from compliance.crypto_shred import CryptoShredVault, ShredReceipt

        return CryptoShredVault if name == "CryptoShredVault" else ShredReceipt
    if name in ("KmsCryptoShredVault", "LocalFileKMS"):
        from compliance.kms_backends import KmsCryptoShredVault, LocalFileKMS

        return (
            KmsCryptoShredVault if name == "KmsCryptoShredVault" else LocalFileKMS
        )
    if name in ("ResidualProof", "prove_vector_erased"):
        from compliance.residual import ResidualProof, prove_vector_erased

        return ResidualProof if name == "ResidualProof" else prove_vector_erased
    if name == "FragmentationBound":
        from compliance.recall_bounds import FragmentationBound

        return FragmentationBound
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
