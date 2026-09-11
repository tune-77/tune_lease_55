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
from fastapi import BackgroundTasks  # noqa: E402

import api.main as main_module  # noqa: E402
from api.schemas import ScoringRequest  # noqa: E402

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
    """FastAPI の直ルートと遅延 include_router が十分に登録されている。"""
    # 公開 method/path の完全性は test_api_route_contract.py の274件署名で検証する。
    # FastAPI の遅延 router は複数APIを1要素として保持するため、ここでは構成の
    # 全消失だけを検知する粗い下限に留める。
    assert len(main_module.app.routes) >= 50


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


def test_calculate_score_exposes_forced_review_reason(monkeypatch):
    result = {
        "score": 88.0,
        "hantei": "要審議",
        "score_based_hantei": "承認圏内",
        "risk_review_required": True,
        "risk_review_reasons": ["Q_risk 強警戒（65.0）"],
        "credit_risk_group_score": 72.0,
        "credit_risk_group_level": "high",
        "credit_risk_group_flags": ["high_q_risk"],
        "quantum_risk": 65.0,
        "q_risk_breakdown": {"total": 65.0},
        "credit_quantum_strong_warning": True,
        "comparison": "test",
        "user_op_margin": 1.0,
        "user_equity_ratio": 20.0,
        "bench_op_margin": 2.0,
        "bench_equity_ratio": 30.0,
        "score_borrower": 90.0,
        "score_base": 88.0,
        "industry_sub": "06 総合工事業",
        "industry_major": "D 建設業",
    }
    monkeypatch.setattr(main_module, "run_quick_scoring", lambda _inputs: result.copy())
    monkeypatch.setattr(main_module, "_record_scoring_memory_usage", lambda *_args: None)

    response = main_module.calculate_score(ScoringRequest(), BackgroundTasks())

    assert response.hantei == "要審議"
    assert response.score_based_hantei == "承認圏内"
    assert response.risk_review_required is True
    assert response.risk_review_reasons == ["Q_risk 強警戒（65.0）"]
    assert response.quantum_risk == 65.0
    assert response.credit_risk_group_level == "high"
