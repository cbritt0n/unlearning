"""
Latent Space Erasure & Graph Healing — Python control plane.

Submodules
----------
main
    FastAPI application (routes, lifespan WAL recovery).
persistence
    Orchestrates WAL BEGIN → C++ mutate → atomic index.bin → WAL COMMIT.
wal
    Append-only binary write-ahead log with SHA-256 record checksums.
"""

__version__ = "0.3.2"
