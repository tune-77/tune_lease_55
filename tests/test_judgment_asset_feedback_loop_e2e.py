"""判断資産フィードバックループを入口から集計まで通しで検査する.

背景（REV-364 / REV-365）: この経路は「配線は全部つながっているのに field_validation が
0 のまま」という壊れ方をしていた。個々の部品のテストは通るのに、部品と部品の継ぎ目で
落ちていたため誰も気づけなかった。壊れていた継ぎ目は2つ:

1. 候補選定が汎用的な正規判断資産を overlap 0 で足切りし、そもそも提示されない
2. レビュー本文の出典（JA-cr-...）記載がLLM任せで、無ければ全部 no_matching_refs で捨てる

このファイルは部品ではなく「継ぎ目」を検査する。守る不変条件は次の1行:

    正規判断資産が候補に出る → 出典が本文に残る → 評価が rule_id へ紐付く
    → field_validation が 0 より大きくなる

途中のどこが切れても、この通しテストが赤くなる。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from api.routers import feedback_loop
from judgment_asset_citation import classify_citations, resolve_rule_ids_from_citations
from scripts.judgment_asset_growth_report import summarize_field_feedback

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_LIB = REPO_ROOT / "frontend" / "src" / "lib"

# フロントの判断資産候補ID は "cr-" + rule_id で、出典は id の先頭8文字で切られる
# （frontend/src/lib/shionReview.ts: formatJudgmentAssetCitation の `item.id.slice(0, 8)`）。
CITATION_ID_SLICE = 8

GENERAL_RULE = {
    "id": "cf61a9701fc8cc42",
    "status": "active",
    "concept": "asset_life_and_residual",
    "canonical_statement": "リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況と満了後の出口を合わせて確認する。",
}
OTHER_RULE = {
    "id": "b259411afb954d6d",
    "status": "active",
    "concept": "business_plan_specificity",
    "canonical_statement": "事業計画は売上見込みだけでなく、受注根拠と返済原資の説明可能性で確認する。",
}


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_lease_data.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    feedback_loop._ensure_shion_screening_reviews_table()
    return db_path


@pytest.fixture
def redirected_logs(tmp_path, monkeypatch):
    usage_log = tmp_path / "judgment_asset_usage_feedback.jsonl"
    drops_log = tmp_path / "judgment_asset_feedback_drops.jsonl"
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_USAGE_FEEDBACK_LOG", usage_log)
    monkeypatch.setattr(feedback_loop, "_JUDGMENT_ASSET_FEEDBACK_DROPS_LOG", drops_log)
    return usage_log, drops_log


@pytest.fixture
def active_rules_file(tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical_judgment_rules.json"
    canonical_path.write_text(
        json.dumps({"rules": [GENERAL_RULE, OTHER_RULE]}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(feedback_loop, "_CANONICAL_JUDGMENT_RULES_JSON", canonical_path)
    return canonical_path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_citation_like_frontend(candidate_id: str, label: str = "正規", topic: str = "screening") -> str:
    """frontend の formatJudgmentAssetCitation と同じ形式の1行を作る."""
    return f"判断資産出典: {label} JA-{candidate_id[:CITATION_ID_SLICE]} / {topic}"


def _insert_review(db_path: Path, *, case_id: str, review_text: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO shion_screening_reviews (case_id, review_text) VALUES (?, ?)",
            (case_id, review_text),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def test_loop_end_to_end_produces_positive_field_validation(
    sqlite_db, redirected_logs, active_rules_file, monkeypatch
):
    """候補選定 → 出典 → 紐付け → 集計 が通しでつながり、field_validation が 0 を超える."""
    usage_log, drops_log = redirected_logs

    # 1. 候補選定: 案件文言と語が重ならない汎用ルールでも候補として提示できる
    monkeypatch.setattr(feedback_loop, "_load_autoresearch_judgment_asset_candidates", lambda *a, **k: [])
    monkeypatch.setattr(feedback_loop, "_load_news_judgment_signals", lambda *a, **k: [])
    selected = feedback_loop._select_screening_judgment_asset_candidates(
        industry_major="建設業",
        industry_sub="土木",
        asset_name="油圧ショベル",
        asset_purpose="更新設備の入替",
        hantei="承認",
        limit=3,
    )
    assert selected, "正規判断資産が1件も候補に出ない（ループの入口が塞がっている）"
    candidate_id = selected[0]["id"]
    assert candidate_id.startswith("cr-")

    # 2. 出典: フロントが本文へ残す形式の1行を、レビュー本文の末尾に持つ
    review_text = "\n".join([
        "違和感",
        "この案件は更新設備の入替だが、旧設備の処分予定が説明と噛み合っていない。",
        "",
        "稟議に残す一文",
        "本件は残価と満了後の出口を確認したうえで判断する。",
        "",
        _format_citation_like_frontend(candidate_id),
    ])
    review_id = _insert_review(sqlite_db, case_id="case-e2e-1", review_text=review_text)

    # 3. 紐付け: レビューへの評価が rule_id 付きで使用ログへ入る
    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    usage_rows = _read_jsonl(usage_log)
    assert usage_rows, "評価が使用ログへ1件も入っていない"
    assert _read_jsonl(drops_log) == [], "出典があるのに drop されている"
    assert {row["rule_id"] for row in usage_rows} == {candidate_id[3:]}
    assert {row["outcome"] for row in usage_rows} == {"helped"}
    assert {row["case_id"] for row in usage_rows} == {"case-e2e-1"}

    # 4. 集計: field_validation が 0 から動く
    rules = [GENERAL_RULE, OTHER_RULE]
    summary = summarize_field_feedback(usage_rows, rules)
    assert summary["score"] > 0, "使用ログがあるのに field_validation が 0 のまま"
    assert summary["totals"]["used"] >= 1
    assert summary["totals"]["helped"] >= 1
    assert summary["totals"]["unknown_rule"] == 0


@pytest.mark.parametrize(
    ("user_feedback", "expected_outcome"),
    [("useful", "helped"), ("thin", "challenged"), ("wrong", "rejected")],
)
def test_user_feedback_maps_to_expected_outcome(
    sqlite_db, redirected_logs, active_rules_file, user_feedback, expected_outcome
):
    """UI の評価ラベルが、集計側が解釈できる outcome へ落ちる."""
    usage_log, _ = redirected_logs
    review_text = "本文\n\n" + _format_citation_like_frontend(f"cr-{GENERAL_RULE['id']}")
    review_id = _insert_review(sqlite_db, case_id="case-map", review_text=review_text)

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, user_feedback)

    rows = _read_jsonl(usage_log)
    assert [row["outcome"] for row in rows] == [expected_outcome]


def test_missing_citation_is_recorded_as_drop_not_silence(sqlite_db, redirected_logs, active_rules_file):
    """出典が無い評価は、黙って消えるのではなく理由付きで drop ログに残る.

    field_validation が 0 のまま動かない時に「押されていない」のか
    「押されたが紐付かなかった」のかを切り分けられる状態を守る。
    """
    usage_log, drops_log = redirected_logs
    review_id = _insert_review(sqlite_db, case_id="case-no-citation", review_text="出典を書かなかったレビュー本文")

    feedback_loop._record_judgment_asset_feedback_from_review(review_id, "useful")

    assert _read_jsonl(usage_log) == [], "紐付かない評価を使用ログへ書いてはいけない"
    drops = _read_jsonl(drops_log)
    assert [row["reason"] for row in drops] == ["no_matching_refs"]
    assert drops[0]["case_id"] == "case-no-citation"


def test_frontend_citation_format_resolves_on_the_python_side():
    """フロントの出典形式と Python 側の解決器が噛み合っていること（言語をまたぐ契約）."""
    citation = _format_citation_like_frontend(f"cr-{GENERAL_RULE['id']}")
    resolved = resolve_rule_ids_from_citations(citation, [GENERAL_RULE["id"], OTHER_RULE["id"]])
    assert resolved == [GENERAL_RULE["id"]]

    # 昇格前の候補（cr- 以外）は判断資産の実績として数えない
    assert resolve_rule_ids_from_citations(
        _format_citation_like_frontend("ar-0123456789abcdef", label="候補"),
        [GENERAL_RULE["id"]],
    ) == []


def test_frontend_still_emits_and_guarantees_citations():
    """出典を出す実装がフロントから消えていないこと.

    過去に ensureCandidateJudgmentAssetMentionInReview がレビュー整理の一環で削除された
    ことがある。出典の保証が同じように消えると、UI は動いたまま field_validation だけが
    静かに 0 へ戻るため、ソースレベルで存在を固定する。
    """
    shion_review = (FRONTEND_LIB / "shionReview.ts").read_text(encoding="utf-8")
    assert "formatJudgmentAssetCitation" in shion_review
    assert "判断資産出典" in shion_review
    assert f"slice(0, {CITATION_ID_SLICE})" in shion_review, "出典IDの切り出し幅が変わっている"
    assert "ensureJudgmentAssetCitations" in shion_review

    # LLM経路は2つある。どちらか片方だけ外れても気づけるように両方を検査する。
    llm_paths = [
        FRONTEND_LIB / "useShionScreeningReview.ts",
        REPO_ROOT / "frontend" / "src" / "app" / "screening" / "page.tsx",
    ]
    for path in llm_paths:
        assert "ensureJudgmentAssetCitations" in path.read_text(encoding="utf-8"), (
            f"{path.name} のLLM経路で出典の保証が外れている"
        )


def test_active_rule_ids_stay_unambiguous_at_citation_truncation():
    """実データの能動ルールが、切り詰めた出典から一意に解決できること.

    出典は rule_id の先頭5文字ほどしか含まない。将来ルールが増えて先頭が衝突すると
    ambiguous になり、評価が黙って捨てられる。実ファイルに対して先に気づけるようにする。
    """
    canonical_path = REPO_ROOT / "data" / "canonical_judgment_rules.json"
    if not canonical_path.exists():
        pytest.skip("canonical_judgment_rules.json が無い環境")
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    active_ids = [
        str(rule.get("id") or "")
        for rule in payload.get("rules") or []
        if isinstance(rule, dict) and rule.get("status") == "active" and rule.get("id")
    ]
    if not active_ids:
        pytest.skip("能動ルールが無い環境")

    for rule_id in active_ids:
        citation = _format_citation_like_frontend(f"cr-{rule_id}")
        result = classify_citations(citation, active_ids)
        assert result["resolved"] == [rule_id], (
            f"rule_id {rule_id} の出典が一意に解決できない（ambiguous={result['ambiguous']}）"
        )
