# Gateway image. Does NOT contain vLLM or CUDA: the model servers run on the
# host next to the GPU, and this container only talks to them over HTTP.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# tectonic compiles LaTeX; it is self-contained, unlike a full TeX Live.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tectonic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer caches across code changes.
COPY pyproject.toml README.md ./
COPY elp ./elp
RUN pip install --upgrade pip && pip install ".[excel]"

COPY config ./config
COPY scripts ./scripts

RUN useradd --create-home --shell /usr/sbin/nologin elp \
    && mkdir -p /app/artifacts/latex \
    && chown -R elp:elp /app
USER elp

EXPOSE 8080

# Four workers keeps the CPU busy on document parsing without oversubscribing
# the single GPU behind the model servers.
CMD ["uvicorn", "elp.main:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", \
     "--proxy-headers", "--forwarded-allow-ips=*", \
     "--timeout-keep-alive", "75"]
