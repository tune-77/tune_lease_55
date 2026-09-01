"""api.main の統合スモークテスト。

api/main.py に残る素のルート・api/routers/ へ抽出済みのルーターを横断して、
「アプリが実際に import・起動でき、代表的な GET エンドポイントが 200 を返す」
ことだけを確認する最小限の回帰ネット。DB 未セットアップの環境でも通る
エンドポイントのみを対象にしており、個々のビジネスロジックの正しさは
別の専用テストが担う。

api/main.py のルーター分割（低リスク群→中リスク群→/api/chat の順）を
進める際、この一覧に含まれるパスの応答が壊れていないかで配線ミスを検知する。
"""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as main_module  # noqa: E402

client = TestClient(main_module.app)

# main.py 本体に残るルートと、api/routers/ へ既に抽出済みのルーターの双方から、
# DB 未セットアップでも 200 を返す代表的な GET エンドポイントを選んでいる。
SMOKE_GET_PATHS = [
    "/health",
    "/healthz",
    "/api/dashboard/stats",
    "/api/improvement-log",
    "/api/vertex-search/widget-config",
    "/api/relationship/state",
    "/api/lease-news/focus",
    "/api/lease-news/brief",
    "/api/lease-news/actions",
    "/api/lease-news/daily-digest",
    "/api/lease-news/recent",
    "/api/lease-intelligence/dialogue/state",
    "/api/lease-intelligence/related-suggestion",
    "/api/chat/history",
    "/api/demo/agent-log",
]


@pytest.mark.parametrize("path", SMOKE_GET_PATHS)
def test_get_endpoint_returns_200(path):
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text[:300]}"


def test_app_has_expected_route_count():
    """ルート数が大きく減っていないか（抽出時の mount 漏れ検知用の粗いガード）。"""
    assert len(main_module.app.routes) >= 60


def test_no_duplicate_method_path_routes():
    """同じHTTPメソッドとパスの二重登録を検知する。"""
    seen = set()
    duplicates = []
    for route in main_module.app.routes:
        path = getattr(route, "path", None)
        for method in getattr(route, "methods", None) or ():
            key = (method, path)
            if key in seen:
                duplicates.append(key)
            seen.add(key)

    assert not duplicates, f"duplicate API routes: {sorted(duplicates)}"


def test_openapi_schema_generation():
    """OpenAPI schema の生成を通常の回帰テストとして保証する。"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
