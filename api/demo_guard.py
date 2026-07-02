"""公開デモ用の書き込み保護ミドルウェア。

ハッカソン等で URL を不特定多数（審査員・来場者）に公開する際、来場者が
`DELETE /api/cases/operation/clear-all`（全案件削除）等でデモデータを破壊できないよう、
環境変数 DEMO_READONLY が有効なときは /api/* への DELETE を 403 で拒否する。

「削除だけ塞ぐ」方針: スコアリング・チャット・討論・案件登録などの試用は許可し、
削除操作のみブロックする（HTTP メソッド DELETE を対象にすることで将来の削除
エンドポイント追加も自動でカバーする）。

DEMO_READONLY 未設定時は完全に無効（ローカル開発・通常運用に無影響）。
api/main.py から独立させているのは重い依存を読み込まずに単体テストするため。
"""
from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_TRUTHY = {"1", "true", "yes", "on"}


def is_demo_readonly() -> bool:
    """DEMO_READONLY が有効かを返す。"""
    return os.environ.get("DEMO_READONLY", "").strip().lower() in _TRUTHY


class DemoReadonlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if is_demo_readonly() and request.method == "DELETE" and request.url.path.startswith("/api/"):
            return Response(
                content='{"detail":"公開デモ環境では削除操作は無効化されています"}',
                status_code=403,
                media_type="application/json",
            )
        return await call_next(request)
