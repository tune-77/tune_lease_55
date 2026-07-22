from lease_intelligence_tools import (
    execute_tool,
    inspect_scoring_policy,
    search_obsidian,
)


def test_inspect_scoring_policy_reports_route_split():
    result = inspect_scoring_policy("物件スコアと借手スコアの統合")

    assert result["status"] == "current_implementation_route_split"
    routes = result["facts"]["routes"]
    assert routes["quick_batch_scoring_core"]["asset_score_affects_final_score"] is False
    assert routes["quick_batch_scoring_core"]["base_score_source"] == "score_borrower"
    assert routes["next_full_api"]["asset_score_affects_final_score"] is True
    assert routes["next_full_api"]["endpoint"] == "/api/score/full"
    assert "経路で異なる" in result["explanation"]


def test_search_obsidian_uses_shared_context_route(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "obsidian_ai_context.collect_obsidian_ai_context",
        lambda *args, **kwargs: {
            "hits": [
                {
                    "path": "Policies/scoring.md",
                    "snippet": "物件評価は警告として扱う。",
                    "source": "keyword",
                    "score": 12.0,
                }
            ]
        },
    )

    result = search_obsidian("asset_score 重み付け", tmp_path)

    assert result["count"] == 1
    assert result["results"][0]["file"] == "Policies/scoring.md"
    assert result["search_route"].startswith("obsidian_query")


def test_execute_tool_dispatches_scoring_policy():
    result = execute_tool(
        "inspect_scoring_policy",
        {"topic": "承認理由"},
    )

    assert result["facts"]["requires_route_identification"] is True


def test_tool_declarations_include_senior_reasoner_contract():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    declaration = next(
        item for item in TOOL_DECLARATIONS
        if item["name"] == "consult_senior_reasoner"
    )
    required = declaration["parameters"]["required"]
    assert required == [
        "question",
        "shion_hypothesis",
        "confidence",
        "evidence_summary",
    ]


def _seed_screening_db(path, rows):
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE screening_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            screened_at TEXT NOT NULL,
            total_score REAL NOT NULL,
            asset_score REAL NOT NULL,
            tenant_score REAL,
            q_risk_score REAL,
            competitor_pressure_score REAL,
            outcome TEXT,
            input_snapshot TEXT,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.executemany(
        "INSERT INTO screening_records (case_id, screened_at, total_score, asset_score, "
        "input_snapshot, source) VALUES (?, ?, ?, ?, ?, 'test')",
        rows,
    )
    conn.commit()
    conn.close()


def test_get_screening_activity_counts_today_with_verdict_breakdown(tmp_path, monkeypatch):
    import datetime
    import json

    import lease_intelligence_tools as tools

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    db = tmp_path / "lease_data.db"
    _seed_screening_db(
        db,
        [
            ("C-1", f"{today.isoformat()}T09:00:00Z", 80.0, 70.0, json.dumps({"company_name": "承認商事"})),
            ("C-2", f"{today.isoformat()}T10:00:00Z", 63.0, 50.0, json.dumps({"company_name": "条件工業"})),
            ("C-3", f"{yesterday.isoformat()}T10:00:00Z", 40.0, 30.0, json.dumps({"company_name": "昨日社"})),
        ],
    )
    monkeypatch.setattr(tools, "DB_PATH", str(db))

    result = tools.get_screening_activity("today")

    assert result["count"] == 2
    assert result["period_label"] == "今日"
    assert result["breakdown"]["承認"] == 1
    assert result["breakdown"]["条件付き承認"] == 1
    assert result["breakdown"]["否決"] == 0
    assert {c["company_name"] for c in result["cases"]} == {"承認商事", "条件工業"}


def test_execute_tool_dispatches_screening_activity(tmp_path, monkeypatch):
    import datetime

    import lease_intelligence_tools as tools

    db = tmp_path / "lease_data.db"
    _seed_screening_db(
        db,
        [("C-1", f"{datetime.date.today().isoformat()}T09:00:00Z", 75.0, 60.0, "{}")],
    )
    monkeypatch.setattr(tools, "DB_PATH", str(db))

    result = tools.execute_tool("get_screening_activity", {"period": "today"})

    assert result["count"] == 1


def test_tool_declarations_include_screening_activity():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    names = {item["name"] for item in TOOL_DECLARATIONS}
    assert "get_screening_activity" in names


def test_get_scoring_coefficients_lists_models_without_args():
    from lease_intelligence_tools import get_scoring_coefficients

    result = get_scoring_coefficients()

    assert "全体_既存先" in result["available_regression_models"]
    assert set(result["coefficient_groups"]) == {"bayesian", "strength_tags", "asset_weight"}


def test_get_scoring_coefficients_returns_named_regression_model():
    from lease_intelligence_tools import get_scoring_coefficients

    result = get_scoring_coefficients("運送業_既存先")

    assert result["model"] == "運送業_既存先"
    assert result["type"] == "regression_coefficients"
    assert "lease_credit_log" in result["coefficients"]


def test_get_scoring_coefficients_bayesian_and_asset_groups():
    from lease_intelligence_tools import get_scoring_coefficients

    bayes = get_scoring_coefficients("bayesian")
    assert bayes["type"] == "bayesian_prior"
    assert "competitor_present" in bayes["coefficients"]

    asset = get_scoring_coefficients("asset_weight")
    assert asset["type"] == "category_asset_obligor_weight"
    first = next(iter(asset["categories"].values()))
    assert "asset_w" in first and "obligor_w" in first


def test_get_scoring_coefficients_feature_across_models():
    from lease_intelligence_tools import get_scoring_coefficients

    result = get_scoring_coefficients(feature="sales_log")

    assert result["type"] == "feature_across_models"
    assert result["count"] >= 2
    assert "全体_既存先" in result["by_model"]


def test_get_scoring_coefficients_unknown_model_returns_candidates():
    from lease_intelligence_tools import get_scoring_coefficients

    result = get_scoring_coefficients("存在しないモデル")

    assert result["found"] is False
    assert "available_regression_models" in result


def test_execute_tool_dispatches_scoring_coefficients():
    from lease_intelligence_tools import execute_tool

    result = execute_tool("get_scoring_coefficients", {"model": "全体_既存先"})

    assert result["type"] == "regression_coefficients"


def test_tool_declarations_include_scoring_coefficients():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    names = {item["name"] for item in TOOL_DECLARATIONS}
    assert "get_scoring_coefficients" in names


def test_obsidian_query_expands_scoring_identifiers_to_business_terms():
    from mobile_app.obsidian_bridge import _expand_query_terms

    terms = _expand_query_terms(
        "scoring_core asset_score score_borrower 統合 重み付け"
    )

    assert "物件スコア" in terms
    assert "借手スコア" in terms
    assert "最終スコア" in terms
    assert "担保価値" in terms
    assert "配点" in terms
