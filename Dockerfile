# syntax=docker/dockerfile:1
# Multi-stage image for the HNSW Healer API.
#
#   docker build -t hnsw-healer:latest .
#   docker run --rm -p 8000:8000 -e HEALER_SIGNING_KEY=... -v healer-data:/app/data hnsw-healer:latest

# ---------------------------------------------------------------------------
# Build: compile the native extension into a wheel
# ---------------------------------------------------------------------------
FROM python:3.11-bookworm AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml setup.py CMakeLists.txt requirements.txt README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY integrations ./integrations
COPY compliance ./compliance

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install cmake ninja "pybind11>=2.12" "numpy>=1.26" \
    && python -m pip wheel . -w /wheels --no-deps -v \
    && python -m pip wheel -r requirements.txt -w /wheels \
    && ls -la /wheels \
    && test -n "$(ls /wheels/hnsw_healer*.whl 2>/dev/null)"

# ---------------------------------------------------------------------------
# Runtime: slim image, non-root
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS release

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HEALER_DATA_DIR=/app/data \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

RUN groupadd --system --gid 10001 healer \
    && useradd --system --uid 10001 --gid healer --home-dir /app --shell /usr/sbin/nologin healer \
    && mkdir -p /app/data \
    && chown -R healer:healer /app

WORKDIR /app

COPY --from=build /wheels /tmp/wheels

# Install project wheel first, then remaining deps (avoid shell glob surprises)
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir /tmp/wheels/hnsw_healer-*.whl \
    && python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && python -c "import hnsw_healer, api.main; print('ok', hnsw_healer.__version__)"

USER healer

VOLUME ["/app/data"]
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
