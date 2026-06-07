# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV FASTAPI_URL=http://127.0.0.1:8000
RUN npm run build


FROM python:3.11-slim-bookworm AS python-deps
WORKDIR /build

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node:20-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=python-deps /opt/venv /opt/venv
COPY . .
COPY .cloudrun_bundle/ /app/.cloudrun_bundle/

RUN rm -rf frontend/.next frontend/node_modules

COPY --from=frontend-builder /build/frontend/.next/standalone/ frontend/
COPY --from=frontend-builder /build/frontend/.next/static/ frontend/.next/static/
COPY --from=frontend-builder /build/frontend/public/ frontend/public/

RUN chmod +x scripts/start_cloud_run.sh \
    && mkdir -p /app/data /app/obsidian_vault /tmp/tune-lease \
    && chmod -R a-w /app/.cloudrun_bundle \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /tmp/tune-lease

USER appuser

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTAPI_HOST=127.0.0.1 \
    FASTAPI_PORT=8000 \
    FASTAPI_URL=http://127.0.0.1:8000 \
    DATA_DIR=/app/data \
    CLOUDRUN_BUNDLE_DIR=/app/.cloudrun_bundle \
    OBSIDIAN_VAULT_PATH=/app/obsidian_vault \
    ENABLE_OBSIDIAN_INDEXING=false \
    ENABLE_FEEDBACK_LOADING=false \
    ENABLE_GUNSHI_RAG=false \
    HOSTNAME=0.0.0.0

EXPOSE 8080

CMD ["./scripts/start_cloud_run.sh"]
