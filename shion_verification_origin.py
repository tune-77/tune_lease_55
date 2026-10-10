"""検証・テストの会話に印を付け、紫苑の記憶・内省の材料から外す（REV-591）。

2026-10-09〜10 に Claude Code の検証（同じ質問の繰り返し・「今日ちょっと疲れた」等の雑談）が
ユーザー本人の会話として記録され、Private Reflection が「ユーザーは現場の重圧で思考停止」
「同じ質問が3回続いたら心理状態を確認する」といった内省・判断資産候補を作った。

印の付け方（どれか1つで検証扱い）:
- HTTP ヘッダー ``X-Shion-Verification: 1``（対話室の Next 経由でも FastAPI まで届く）
- 検証用ユーザーID（``rev544_verify`` / ``verify_rev532`` のように verify / verification を区切りで含む）
- 呼び出し側スクリプトは ``verification_client_headers()`` を付ける（``AI_LIVE_VERIFY=1`` か
  ``SHION_VERIFICATION=1`` の時に自動でヘッダーが付く）
- 検証用に ``AI_LIVE_VERIFY=1`` / ``SHION_VERIFICATION=1`` で起動したサーバーは、全部の会話を検証扱いにする

検証の会話も返答は通常どおり作り、会話ログには ``origin: verification`` を付けて残す。
止めるのは記憶・内省・関係性・予想・要点・Knowledge・判断資産候補への書き込みと、
夜間の内省・成長記録・イラストの題材として読むことだけ。
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from typing import Any, Mapping

VERIFICATION_HEADER = "X-Shion-Verification"
VERIFICATION_ORIGIN = "verification"
VERIFICATION_HISTORY_SUFFIX = ":verification"
# Obsidian の対話ノートで検証の往復の見出しに付ける印（内省はこの節を読まない）
VERIFICATION_NOTE_MARK = "〔検証・本人の会話ではない〕"
# 検証の会話を本人の会話と読んだ内省ノートに、冒頭（見出しの直後）へ足す訂正の節。次の内省は本文の代わりにこれを読む
VERIFICATION_CORRECTION_HEADING = "## 訂正（検証の会話）"

_TRUTHY = {"1", "true", "yes", "on", VERIFICATION_ORIGIN}
_VERIFY_USER_ID = re.compile(r"(?:^|[_\-:.])verif(?:y|ication)(?:$|[_\-:.])", re.IGNORECASE)
_HEADER_KEY = VERIFICATION_HEADER.lower().encode("latin-1")

_turn_origin: ContextVar[str] = ContextVar("shion_turn_origin", default="")


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def is_verification_user_id(user_id: Any) -> bool:
    return bool(_VERIFY_USER_ID.search(str(user_id or "")))


def mark_verification_turn(*, user_id: Any = "", header_value: Any = "") -> bool:
    """この会話（リクエスト）を検証扱いにする。印が無ければ何もしない。現在の状態を返す。"""
    if _truthy(header_value) or is_verification_user_id(user_id):
        _turn_origin.set(VERIFICATION_ORIGIN)
    return is_verification_turn()


def is_verification_turn() -> bool:
    return _turn_origin.get() == VERIFICATION_ORIGIN


def origin_fields() -> dict[str, str]:
    """会話ログの行に足す印。通常の会話では空（既存の行の形を変えない）。"""
    return {"origin": VERIFICATION_ORIGIN} if is_verification_turn() else {}


def history_user_id(user_id: str) -> str:
    """検証の会話は画面・次回の会話履歴を本人の会話と分ける。"""
    user_id = str(user_id or "default")
    if not is_verification_turn() or user_id.endswith(VERIFICATION_HISTORY_SUFFIX):
        return user_id
    return user_id + VERIFICATION_HISTORY_SUFFIX


def is_verification_row(row: Any) -> bool:
    """会話ログ・経験ログの1行が検証由来か（記録済みデータの印・検証用ユーザーIDのどちらでも）。"""
    if not isinstance(row, Mapping):
        return False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    if VERIFICATION_ORIGIN in (row.get("origin"), metadata.get("origin")):
        return True
    if str(row.get("call_class") or "").strip().lower() == VERIFICATION_ORIGIN:
        return True
    return is_verification_user_id(row.get("user_id"))


def verification_client_headers(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """検証スクリプトが紫苑の API を呼ぶ時に付けるヘッダー。"""
    env = os.environ if env is None else env
    if _truthy(env.get("AI_LIVE_VERIFY")) or _truthy(env.get("SHION_VERIFICATION")):
        return {VERIFICATION_HEADER: "1"}
    return {}


class VerificationOriginMiddleware:
    """ヘッダー（またはサーバーの検証用起動）の印をリクエストの間だけ有効にする。

    同期エンドポイントはこの文脈の写しで動き、背景処理（api/background_executor）にも引き継がれる。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        value = next((v for k, v in scope.get("headers") or [] if k.lower() == _HEADER_KEY), b"")
        verification = _truthy(value.decode("latin-1")) or bool(verification_client_headers())
        token = _turn_origin.set(VERIFICATION_ORIGIN if verification else "")
        try:
            await self.app(scope, receive, send)
        finally:
            _turn_origin.reset(token)
