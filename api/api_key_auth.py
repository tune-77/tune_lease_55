"""共有シークレットによる API アクセス制御ミドルウェア（多層防御）。

環境変数 API_ACCESS_KEY が設定されている時のみ有効化され、/api/* への全リクエストに
一致する X-API-Key ヘッダ（または Authorization: Bearer <key>）を要求する。
未設定時は完全に無効（ローカル開発・テスト・Next.js rewrite 構成を壊さない）。

主防御は Cloud Run IAM（--no-allow-unauthenticated）だが、無認証公開時や
トンネル公開時の保険として API 自身でも拒否できるようにする。詳細は CLOUD_RUN.md。

api/main.py から独立させているのは、重い依存を読み込まずに単体テストできるようにするため。
"""
from __future__ import annotations

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# 認証不要の公開パス（ヘルスチェック・ルート・APIドキュメント）
AUTH_EXEMPT_PATHS = {"/", "/healthz", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


def get_api_access_key() -> str:
    """設定された API アクセスキー（未設定なら空文字）を返す。"""
    return os.environ.get("API_ACCESS_KEY", "").strip()


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        access_key = get_api_access_key()
        if not access_key:
            return await call_next(request)
        path = request.url.path
        # CORS プリフライトと公開パスは検証しない
        if request.method == "OPTIONS" or path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        # 保護対象は API パスのみ（静的・ドキュメント配信は上で除外済み）
        if not path.startswith("/api/"):
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if not provided:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided = auth_header[len("Bearer "):]
        # タイミング攻撃を避けるため定数時間比較
        if not (provided and hmac.compare_digest(provided, access_key)):
            return Response(
                content='{"detail":"API キー認証に失敗しました"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
