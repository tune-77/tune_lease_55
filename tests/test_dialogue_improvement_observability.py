"""Phase 0（紫苑中心・改善ループ移行計画）: 対話室の改善ループ観測の2層コンテキスト。

- 常時レイヤ: 異常サマリ1〜2行のみ。異常がなければ空（通常会話の注入量を増やさない）
- オンデマンドレイヤ: 改善相談のメッセージのときだけ詳細を遅延ロード
- すべて read-only・ファイル欠損に耐える
"""

import json

import pytest


@pytest.fixture()
def main_module(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_REPO_ROOT", str(tmp_path))
    return main


def _write_pipeline_log(tmp_path, rows):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "pipeline_step_log.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normal_message_injects_nothing_when_healthy(tmp_path, main_module):
    _write_pipeline_log(
        tmp_path,
        [
            {"ts": "2026-07-17T04:00:00Z", "run_date": "20260717", "step": "build_shion_memory_index", "exit_code": 0},
            {"ts": "2026-07-17T04:01:00Z", "run_date": "20260717", "step": "auto_improvement_pipeline", "exit_code": 0},
        ],
    )

    context = main_module._build_dialogue_improvement_observability_context("この会社の審査どう思う？")

    assert context == ""


def test_anomaly_summary_is_single_line_for_normal_message(tmp_path, main_module):
    _write_pipeline_log(
        tmp_path,
        [
            {"ts": "2026-07-16T04:00:00Z", "run_date": "20260716", "step": "gist_update", "exit_code": 1},
            {"ts": "2026-07-17T04:00:00Z", "run_date": "20260717", "step": "eval_shion_memory_recall", "exit_code": 2},
            {"ts": "2026-07-17T04:01:00Z", "run_date": "20260717", "step": "sync_improvement_reports", "exit_code": 1},
            {"ts": "2026-07-17T04:02:00Z", "run_date": "20260717", "step": "build_codex_auto_queue", "exit_code": 0},
        ],
    )

    context = main_module._build_dialogue_improvement_observability_context("こんにちは、元気？")

    assert "2ステップが失敗" in context
    assert "eval_shion_memory_recall" in context
    assert "\n" not in context  # 常時レイヤは1行のみ
    assert "改善ループ観測・詳細" not in context
    # 最新 run_date のみが対象（前日の gist_update 失敗は数えない）
    assert "gist_update" not in context


def test_rerun_of_same_step_uses_last_result(tmp_path, main_module):
    _write_pipeline_log(
        tmp_path,
        [
            {"ts": "2026-07-17T04:00:00Z", "run_date": "20260717", "step": "auto_improvement_pipeline", "exit_code": 1},
            {"ts": "2026-07-17T05:00:00Z", "run_date": "20260717", "step": "auto_improvement_pipeline", "exit_code": 0},
        ],
    )

    assert main_module._build_pipeline_anomaly_summary_line() == ""


def test_consultation_message_expands_pipeline_detail(tmp_path, main_module):
    _write_pipeline_log(
        tmp_path,
        [
            {"ts": "2026-07-17T04:00:00Z", "run_date": "20260717", "step": "sync_cloudrun_inputs_from_gcs", "exit_code": 1},
            {"ts": "2026-07-17T04:01:00Z", "run_date": "20260717", "step": "build_shion_memory_index", "exit_code": 0},
        ],
    )

    context = main_module._build_dialogue_improvement_observability_context(
        "昨夜のパイプラインで失敗したステップは？"
    )

    assert "改善ループ観測・詳細" in context
    assert "sync_cloudrun_inputs_from_gcs(exit 1)" in context
    assert "全2ステップ中1件失敗" in context
    # 権限を装わないガード文言
    assert "実行権限を持っているように装わない" in context


def test_ledger_summary_last_entry_wins(tmp_path, main_module):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    rows = [
        {"key": "misc_aaa", "canonical_key": "misc_aaa", "rev_id": "REV-214", "status": "parked",
         "title": "対話室の改善", "recorded_at": "2026-07-10T01:00:00"},
        {"key": "misc_aaa", "canonical_key": "misc_aaa", "rev_id": "REV-214", "status": "applied",
         "title": "対話室の改善", "pr_url": "https://example.com/pr/1", "recorded_at": "2026-07-15T01:00:00"},
        {"key": "misc_bbb", "canonical_key": "misc_bbb", "rev_id": "REV-215", "status": "rejected",
         "title": "却下された案", "recorded_at": "2026-07-14T01:00:00"},
    ]
    (scripts_dir / "improvement_ledger.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    context = main_module._build_dialogue_improvement_observability_context("今週マージされたREVは？")

    assert "REV-214" in context
    assert "applied:1" in context
    assert "rejected:1" in context
    assert "parked" not in context  # 同一キーは最後のエントリ(applied)が有効


def test_codex_queue_and_recursive_digest_in_detail(tmp_path, main_module):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "codex_auto_queue_20260717.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-17T04:05:00",
                "status": "READY",
                "queued_count": 1,
                "items": [{"id": "REV-216", "title": "表示文言の修正"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "codex_auto_execution_status.json").write_text(
        json.dumps(
            {"items": {"REV-216": {"id": "REV-216", "status": "completed_pending_review", "attempts": 1}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports_dir / "recursive_self_improvement_latest.md").write_text(
        "# Recursive Self-Improvement Report\n\n- Ranked queue: 0\n- Suppressed: 36\n",
        encoding="utf-8",
    )

    context = main_module._build_dialogue_improvement_observability_context("Codexキューの状況を教えて")

    assert "REV-216" in context
    assert "completed_pending_review" in context
    assert "Ranked queue: 0" in context


def test_pm_quality_summary_in_detail(tmp_path, main_module):
    """P3-2: 事後検証レポートの的中率・Overrule率が改善相談の詳細に載る。"""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "shion_pm_quality_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-18T04:10:00",
                "kpis": {
                    "triage_total": 6,
                    "hit_rates_by_classifier": {
                        "user": {"resolved": 4, "applied": 3, "hit_rate": 0.75},
                    },
                    "overrule": {"with_rule_decision": 5, "overruled": 1, "rate": 0.2},
                    "lead_time_days_avg": 2.5,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    context = main_module._build_dialogue_improvement_observability_context("トリアージの的中率は？")

    assert "トリアージ事後検証" in context
    assert "user 3/4 (75%)" in context
    assert "Overrule率 20%" in context
    assert "判断→マージ平均 2.5日" in context
    assert "数字が無い項目は計測前と言う" in context


def test_detail_survives_missing_files(tmp_path, main_module):
    # data/ も scripts/ も reports/ も無い環境（本セッションのチェックアウトと同じ）でも例外を出さない
    context = main_module._build_dialogue_improvement_observability_context("改善候補の相談をしたい")

    assert "改善ループ観測・詳細" in context
    assert "確認できない" in context
