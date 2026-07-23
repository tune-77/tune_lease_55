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


def test_get_pipeline_status_aggregates_ledger_report_and_tasks(tmp_path, monkeypatch):
    import json

    import lease_intelligence_tools as tools

    # 疑似リポジトリ構造を用意
    (tmp_path / "scripts").mkdir()
    (tmp_path / "reports").mkdir()
    ledger = tmp_path / "scripts" / "improvement_ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"canonical_key": "k1", "rev_id": "REV-001", "status": "needs_review", "title": "A"}),
                # k1 は後で applied に更新（最後が有効）
                json.dumps({"canonical_key": "k1", "rev_id": "REV-001", "status": "applied", "title": "A",
                            "recorded_at": "2026-07-20T10:00:00", "pr_url": "http://pr/1"}),
                json.dumps({"canonical_key": "k2", "rev_id": "REV-002", "status": "applied", "title": "B",
                            "recorded_at": "2026-07-21T10:00:00", "pr_url": "http://pr/2"}),
                json.dumps({"canonical_key": "k3", "rev_id": "REV-003", "status": "rejected", "title": "C"}),
            ]
        ),
        encoding="utf-8",
    )
    report = tmp_path / "reports" / "recursive_self_improvement_latest.md"
    report.write_text(
        "# Recursive Self-Improvement Report\n\n"
        "- Generated at: `2026-07-21T04:01:58`\n"
        "- Canonical candidates: 34\n"
        "- Ranked queue: 0\n"
        "- Suppressed: 34\n\n"
        "## Measurement\n"
        "- PDCA rate: 100.0%\n"
        "- Noise rate: 12.5%\n",
        encoding="utf-8",
    )
    tasks = tmp_path / "shion_pending_tasks.json"
    tasks.write_text(
        json.dumps([
            {"id": "1", "topic": "残価テーブル調査", "status": "pending"},
            {"id": "2", "topic": "済んだ調査", "status": "done"},
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_REPO_PATH", tmp_path)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    result = tools.get_pipeline_status()

    ledger_summary = result["rev_ledger"]
    assert ledger_summary["available"] is True
    assert ledger_summary["total_revs"] == 3
    assert ledger_summary["status_counts"] == {"applied": 2, "rejected": 1}
    # 最新の applied が先頭（REV-002, 2026-07-21）
    assert ledger_summary["recent_applied"][0]["rev_id"] == "REV-002"

    report_summary = result["self_improvement_report"]
    assert report_summary["available"] is True
    assert report_summary["generated_at"] == "2026-07-21T04:01:58"
    assert report_summary["suppressed"] == "34"
    assert report_summary["metrics"]["pdca_rate"] == "100.0%"
    assert report_summary["metrics"]["noise_rate"] == "12.5%"

    tasks_summary = result["pending_investigations"]
    assert tasks_summary["available"] is True
    assert tasks_summary["open_count"] == 1
    assert tasks_summary["total_count"] == 2
    assert tasks_summary["open_topics"] == ["残価テーブル調査"]


def test_get_pipeline_status_handles_missing_sources(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    monkeypatch.setattr(tools, "_REPO_PATH", tmp_path)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    result = tools.get_pipeline_status()

    assert result["rev_ledger"]["available"] is False
    assert result["self_improvement_report"]["available"] is False
    assert result["pending_investigations"]["available"] is False


def test_get_pipeline_status_excludes_stale_and_expired_tasks(tmp_path, monkeypatch):
    """未完了件数は「陳腐化していない pending」のみ。放置された古い約束や expired は除外。"""
    import json

    import lease_intelligence_tools as tools

    tasks = tmp_path / "shion_pending_tasks.json"
    tasks.write_text(
        json.dumps([
            {"id": "1", "topic": "生きてる約束", "status": "pending", "promised_at": "2999-01-01T00:00:00"},
            {"id": "2", "topic": "放置された古い約束", "status": "pending", "promised_at": "2020-01-01T00:00:00"},
            {"id": "3", "topic": "期限切れ", "status": "expired"},
            {"id": "4", "topic": "完了済み", "status": "done"},
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_REPO_PATH", tmp_path)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    summary = tools.get_pipeline_status()["pending_investigations"]

    assert summary["available"] is True
    assert summary["total_count"] == 4
    assert summary["open_count"] == 1
    assert summary["expired_count"] == 1
    assert summary["open_topics"] == ["生きてる約束"]


def test_get_pipeline_status_reads_report_from_bundle_json(tmp_path, monkeypatch):
    """Cloud Run 相当: reports/ に .md が無く、バンドルに .json だけある状況。

    .md 直読みだけだと available=False（「レポート未生成」誤報告）になる回帰を防ぐ。
    """
    import json

    import lease_intelligence_tools as tools

    # トップレベル reports/ は空（Cloud Run では .dockerignore で除外される想定）
    (tmp_path / "reports").mkdir()

    bundle_reports = tmp_path / ".cloudrun_bundle" / "reports"
    bundle_reports.mkdir(parents=True)
    (bundle_reports / "recursive_self_improvement_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-23T04:01:58",
                "canonical_candidate_count": 34,
                "ranked_queue_count": 2,
                "suppressed_count": 32,
                "measurement_summary": {
                    "pdca_rate": 100.0,
                    "response_changed_rate": 36.7,
                    "repeat_issue_rate": 0.0,
                    "reuse_rate": 100.0,
                    "noise_rate": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_REPO_PATH", tmp_path)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.delenv("CLOUDRUN_BUNDLE_DIR", raising=False)

    report_summary = tools.get_pipeline_status()["self_improvement_report"]

    assert report_summary["available"] is True
    assert report_summary["source"] == "json"
    assert report_summary["generated_at"] == "2026-07-23T04:01:58"
    assert report_summary["canonical_candidates"] == "34"
    assert report_summary["ranked_queue"] == "2"
    assert report_summary["suppressed"] == "32"
    assert report_summary["metrics"]["pdca_rate"] == "100.0%"
    assert report_summary["metrics"]["noise_rate"] == "12.5%"


def test_execute_tool_dispatches_pipeline_status(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    monkeypatch.setattr(tools, "_REPO_PATH", tmp_path)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    result = tools.execute_tool("get_pipeline_status", {"recent": 3})

    assert "rev_ledger" in result


def test_tool_declarations_include_pipeline_status():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    names = {item["name"] for item in TOOL_DECLARATIONS}
    assert "get_pipeline_status" in names


def _seed_portfolio_db(path):
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE past_cases (id TEXT PRIMARY KEY, timestamp TEXT, industry_sub TEXT, "
        "score REAL, user_eq REAL, final_status TEXT, data TEXT)"
    )
    conn.execute("CREATE TABLE screening_records (id INTEGER PRIMARY KEY, case_id TEXT)")
    conn.executemany(
        "INSERT INTO past_cases (id, timestamp, industry_sub, score, final_status) VALUES (?,?,?,?,?)",
        [
            ("1", "2026-01-01T00:00:00Z", "建設", 70.0, "成約"),
            ("2", "2026-02-01T00:00:00Z", "建設", 60.0, "成約"),
            ("3", "2026-03-01T00:00:00Z", "運輸", 50.0, "失注"),
        ],
    )
    conn.executemany(
        "INSERT INTO screening_records (case_id) VALUES (?)", [("1",), ("2",)]
    )
    conn.commit()
    conn.close()


def test_get_system_overview_reports_models_thresholds_and_data(tmp_path, monkeypatch):
    import json

    import lease_intelligence_tools as tools

    (tmp_path / "training_meta.json").write_text(
        json.dumps({"last_trained_count": 1922, "last_trained_at": "2026-06-21", "last_auc": 0.68, "total_runs": 13}),
        encoding="utf-8",
    )
    (tmp_path / "ensemble_config.json").write_text(
        json.dumps({"ensemble_alpha": 0.4, "auc_ensemble": 0.82}), encoding="utf-8"
    )
    (tmp_path / "quantum_config.json").write_text(
        json.dumps({"thresholds": {"secondary_review": 35.0, "high_risk": 60.0}}), encoding="utf-8"
    )
    (tmp_path / "business_rules.json").write_text(
        json.dumps({"thresholds": {"approval": 0.77, "review": 0.4}}), encoding="utf-8"
    )
    db = tmp_path / "lease_data.db"
    _seed_portfolio_db(db)

    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(tools, "DB_PATH", str(db))

    result = tools.get_system_overview()

    assert result["models"]["last_auc"] == 0.68
    assert result["thresholds"]["score_approval_line"] == 71
    assert result["thresholds"]["q_risk_high_risk"] == 60.0
    assert result["thresholds"]["probability_approval"] == 0.77
    assert result["data"]["past_cases"] == 3
    assert result["data"]["screening_records"] == 2


def test_lookup_judgment_rules_aggregates_three_sources(tmp_path, monkeypatch):
    import json

    import lease_intelligence_tools as tools

    (tmp_path / "canonical_judgment_rules.json").write_text(
        json.dumps({
            "generated_at": "2026-07-18",
            "rules": [
                {"id": "r1", "concept": "cash_flow", "canonical_statement": "返済原資を確認する", "status": "active", "confidence": 0.9},
                {"id": "r2", "concept": "asset", "canonical_statement": "担保価値を評価する", "status": "active", "confidence": 0.8},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "business_rules.json").write_text(
        json.dumps({
            "thresholds": {"approval": 0.77},
            "custom_rules": [{"name": "債務超過は要審議", "industry": "ALL", "action_type": "force_status", "action_value": "要審議", "conditions": []}],
        }),
        encoding="utf-8",
    )
    (tmp_path / "secondary_review_items.json").write_text(
        json.dumps([
            {"id": "sr001", "category": "財務確認", "text": "決算書確認", "required": True},
            {"id": "sr002", "category": "信用調査", "text": "信用照会", "required": False},
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    result = tools.lookup_judgment_rules()
    assert result["canonical_rules"]["total"] == 2
    assert result["business_rules"]["custom_rules"][0]["name"] == "債務超過は要審議"
    assert result["secondary_review"]["by_category"] == {"財務確認": 1, "信用調査": 1}
    assert result["secondary_review"]["required"] == 1

    # query で canonical 本文を絞り込める
    filtered = tools.lookup_judgment_rules(query="返済原資")
    assert filtered["canonical_rules"]["matched"] == 1
    assert filtered["canonical_rules"]["rules"][0]["id"] == "r1"


def test_get_industry_benchmark_lists_and_matches(tmp_path, monkeypatch):
    import json

    import lease_intelligence_tools as tools

    (tmp_path / "industry_benchmarks.json").write_text(
        json.dumps({"06 総合工事業": {"op_margin": 2.9, "equity_ratio": 47.4}}),
        encoding="utf-8",
    )
    (tmp_path / "industry_trends.json").write_text(
        json.dumps({"_comment": "x", "建設業": "建設業のトレンド本文"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))

    listing = tools.get_industry_benchmark()
    assert "06 総合工事業" in listing["available_benchmarks"]
    assert "建設業" in listing["available_trend_sectors"]

    matched = tools.get_industry_benchmark("総合工事業")
    assert matched["found"] is True
    assert matched["benchmarks"]["06 総合工事業"]["equity_ratio"] == 47.4

    trend = tools.get_industry_benchmark("建設業")
    assert trend["trends"]["建設業"] == "建設業のトレンド本文"


def test_get_portfolio_stats_computes_contract_rate(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    db = tmp_path / "lease_data.db"
    _seed_portfolio_db(db)
    monkeypatch.setattr(tools, "DB_PATH", str(db))

    result = tools.get_portfolio_stats()

    assert result["past_cases_total"] == 3
    assert result["outcome_distribution"] == {"成約": 2, "失注": 1}
    assert result["contract_rate_pct"] == 66.7
    assert result["top_industries"][0]["industry"] == "建設"
    assert result["screening_records_total"] == 2


def test_execute_tool_dispatches_new_overview_tools(tmp_path, monkeypatch):
    import lease_intelligence_tools as tools

    db = tmp_path / "lease_data.db"
    _seed_portfolio_db(db)
    monkeypatch.setattr(tools, "get_data_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(tools, "DB_PATH", str(db))

    assert "thresholds" in tools.execute_tool("get_system_overview", {})
    assert "canonical_rules" in tools.execute_tool("lookup_judgment_rules", {})
    assert "available_benchmarks" in tools.execute_tool("get_industry_benchmark", {})
    assert tools.execute_tool("get_portfolio_stats", {})["past_cases_total"] == 3


def test_tool_declarations_include_overview_tools():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    names = {item["name"] for item in TOOL_DECLARATIONS}
    assert {"get_system_overview", "lookup_judgment_rules",
            "get_industry_benchmark", "get_portfolio_stats"} <= names


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
