# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

COPY frontend/ ./
ENV FASTAPI_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_HIDE_RESEARCH_ORGAN=1
RUN npm run build && npm cache clean --force


FROM python:3.11-slim-bookworm AS python-deps
WORKDIR /build

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_CACHE=1

RUN pip install --no-cache-dir --upgrade pip uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# REV-158: psycopg2-binary を venv に追加（uv.lock を変更せずに uv pip で直接インストール）
# uv sync が作る venv には pip バイナリが含まれないため uv pip を使用
# Cloud SQL (PostgreSQL) 接続に必要。ローカル開発では DATABASE_URL 未設定のため実行されない。
RUN uv pip install --python /opt/venv/bin/python --no-cache-dir "psycopg2-binary>=2.9.0" && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete


FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# 最小限のシステムライブラリのみインストール（マルチステージビルドで curl,git不要）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=node:20-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=python-deps /opt/venv /opt/venv
COPY . .
COPY .cloudrun_bundle/ /app/.cloudrun_bundle/

# 不要なファイル削除：フロントエンド・Python キャッシュ
RUN rm -rf \
    frontend/.next frontend/node_modules \
    .git .github __pycache__ \
    tests models/sentence-transformers \
    api/chroma_db/* obsidian_vault/* && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.dist-info" \) -delete

COPY --from=frontend-builder /build/frontend/.next/standalone/ frontend/
COPY --from=frontend-builder /build/frontend/.next/static/ frontend/.next/static/
COPY --from=frontend-builder /build/frontend/public/ frontend/public/

RUN chmod +x scripts/start_cloud_run.sh scripts/entrypoint.sh \
    && mkdir -p /app/data /app/data-git /app/api/chroma_db /app/obsidian_vault /tmp/tune-lease \
    && chmod -R a-w /app/.cloudrun_bundle \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /tmp/tune-lease

USER appuser

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
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

CMD ["./scripts/entrypoint.sh"]
