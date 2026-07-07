# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked


FROM python:3.13-slim

RUN groupadd --system --gid 999 mcps \
 && useradd --system --gid 999 --uid 999 --create-home mcps

WORKDIR /app

COPY --from=builder --chown=mcps:mcps /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    VAULT=/vault \
    CONTAINER=true \
    FASTMCP_CLIENT_LOG_LEVEL=warning

USER mcps

EXPOSE 8000

CMD ["mcps", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
