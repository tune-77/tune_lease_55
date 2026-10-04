"""Cloudflare Workers AI 上の Clef / Clef-flash を、Jev（TypeSafe System One）と同じ形で呼ぶ。

Clef は System One と互換（state・questions・answers の形が同じ）。違いは送り先と、応答が
Cloudflare の封筒 {"result": ..., "success": ...} に包まれる点だけなので、ここで剥がして
Jev の応答と同じ dict を返す。API トークンは環境変数 CLOUDFLARE_API_TOKEN からのみ読み、表示しない。
"""

from __future__ import annotations

import os
from typing import Any, Callable

import httpx

import typesafe_dedup_guard as transport

API_BASE = "https://api.cloudflare.com/client/v4"
ZONE_NAME = "tune77.com"  # account_id はこのゾーンの持ち主から取る

# Jev は既存の評価・本番と同じ送信口（ハード期限つき）を使う。
jev_request: Callable[[dict], dict] = transport._default_request


class ClefError(RuntimeError):
    pass


class ClefClient:
    def __init__(self, token: str, account_id: str, *, timeout: float = 90.0):
        self._token = token
        self.account_id = account_id
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"ClefClient(account_id={self.account_id!r}, token=<hidden>)"

    @classmethod
    def from_env(cls) -> "ClefClient":
        token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not token:
            raise ClefError("CLOUDFLARE_API_TOKEN is not set")
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            response = httpx.get(f"{API_BASE}/zones", params={"name": ZONE_NAME},
                                 headers={"Authorization": f"Bearer {token}"}, timeout=30)
            zones = response.json().get("result") or []
            if not zones:
                raise ClefError(f"zone {ZONE_NAME} not visible to the token")
            account_id = zones[0]["account"]["id"]
        return cls(token, account_id)

    def request(self, payload: dict[str, Any], model: str) -> dict[str, Any]:
        body = {**payload, "model": model}
        response = httpx.post(
            f"{API_BASE}/accounts/{self.account_id}/ai/run/@cf/cloudflare/{model}",
            headers={"Authorization": f"Bearer {self._token}"}, json=body, timeout=self.timeout,
        )
        try:
            envelope = response.json()
        except ValueError:
            raise ClefError(f"HTTP {response.status_code}: non-JSON response") from None
        if response.status_code != 200 or not envelope.get("success", False):
            errors = envelope.get("errors") or envelope
            raise ClefError(f"HTTP {response.status_code}: {str(errors)[:300]}")
        result = envelope.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ClefError("response is missing result.answers")
        return {**result, "model": result.get("model") or model}

    def sender(self, model: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return lambda payload: self.request(payload, model)
