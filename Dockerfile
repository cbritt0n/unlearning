# syntax=docker/dockerfile:1.6
# =============================================================================
# Multi-stage production image for Latent Space Erasure & Graph Healing API
# =============================================================================
#
# Stage 1 (build)  — full C++ toolchain; compile hnsw_healer into a wheel
# Stage 2 (release)— slim runtime; unprivileged uvicorn serving api.main:app
#
# Build:
#   docker build -t hnsw-healer:latest .
# Run:
#   docker run --rm -p 8000:8000 -v healer-data:/app/data hnsw-healer:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — compile the pybind11 extension into a binary wheel
# ---------------------------------------------------------------------------
FROM python:3.11-bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# CMake + g++ for the native hnsw_healer module
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Install build-system deps first for better layer caching
COPY pyproject.toml setup.py CMakeLists.txt requirements.txt ./
COPY src ./src
COPY api ./api
COPY integrations ./integrations
COPY compliance ./compliance

RUN python -m pip install --upgrade pip setuptools wheel build \
    && python -m pip install cmake ninja pybind11 numpy \
    && python -m pip wheel . -w /wheels --no-deps \
    && python -m pip wheel -r requirements.txt -w /wheels

# ---------------------------------------------------------------------------
# Stage 2: Release — minimal image, wheel only, non-root process
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS release

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persistence paths (WAL + index.bin)
    HEALER_DATA_DIR=/app/data \
    # Bind all interfaces inside the container
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Unprivileged system user (no login shell, fixed UID for K8s runAsNonRoot)
RUN groupadd --system --gid 10001 healer \
    && useradd --system --uid 10001 --gid healer --home-dir /app --shell /usr/sbin/nologin healer \
    && mkdir -p /app/data \
    && chown -R healer:healer /app

WORKDIR /app

# Copy prebuilt wheels from the build stage (compiled .so + pure-Python deps)
COPY --from=build /wheels /tmp/wheels

# Install the native wheel + API requirements; discard wheel cache afterward
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && python -c "import hnsw_healer; import api.main; print('hnsw_healer', hnsw_healer.__version__)"

USER healer

VOLUME ["/app/data"]
EXPOSE 8000

# Production ASGI server — single worker; scale out with replicas / gunicorn if needed
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
