"""正規判断資産が審査画面の候補選定で足切りされないことの検査.

背景（REV-364）: 候補選定には「案件文言との単語overlapが0で、かつ useful_count /
edit_count も0なら落とす」足切りがある。正規判断資産は
_load_canonical_judgment_asset_candidates() が useful_count / edit_count を常に0で
読み込むため、この2つの逃げ道を自力で満たせない。結果として

    overlap 0 → 候補に出ない → 引用されない → 評価が付かない
             → useful_count 0 のまま → また overlap 0 で足切り

という閉ループになり、能動ルールが永久に眠って field_validation が0のままだった。
"""
from __future__ import annotations

import pytest

from api.routers import feedback_loop


def _general_canonical_rule() -> dict:
    """案件文言と語がまったく重ならない、汎用的に書かれた正規判断資産."""
    return {
        "id": "cr-abcdef0123456789",
        "source": "canonical_judgment_rules",
        "candidate_type": "application_rule",
        "research_topic": "canonical_judgment_rule",
        "claim": "条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。",
        "effective_claim": "条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。",
        "promotion_status": "active",
        "verified_status": "canonical",
        "use_count": 0,
        "useful_count": 0,
        "edit_count": 0,
        "rejected_count": 0,
    }


def _unrelated_context_terms() -> set[str]:
    return feedback_loop._screening_candidate_terms("建設業", "土木", "油圧ショベル", "更新設備の入替", "承認")


def test_general_canonical_rule_has_no_overlap_but_positive_score():
    """前提の確認: 汎用的な正規判断資産は overlap 0 でもスコアは正になる."""
    rule = _general_canonical_rule()
    score, overlap, _, _ = feedback_loop._rank_screening_judgment_asset_candidate(
        rule, _unrelated_context_terms()
    )
    assert overlap == 0
    assert score > 0, "canonical_bonus によりスコア自体は正のはず"


def test_canonical_rule_survives_overlap_cut(monkeypatch, tmp_path):
    """overlap 0 でも正規判断資産は候補に残る（閉ループを断つ）."""
    monkeypatch.setattr(
        feedback_loop, "_load_canonical_judgment_asset_candidates", lambda *a, **k: [_general_canonical_rule()]
    )
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
    assert [item["id"] for item in selected] == ["cr-abcdef0123456789"]


def test_non_canonical_candidate_is_still_cut_when_unrelated(monkeypatch):
    """足切りの緩和は正規判断資産だけ。無関係な通常候補は従来どおり落とす."""
    unrelated = {
        "id": "ar-0123456789abcdef",
        "source": "autoresearch_judgment_asset_candidates",
        "candidate_type": "application_rule",
        "research_topic": "unrelated",
        "claim": "条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。",
        "promotion_status": "not_promoted",
        "verified_status": "unverified",
        "use_count": 0,
        "useful_count": 0,
        "edit_count": 0,
        "rejected_count": 0,
    }
    monkeypatch.setattr(feedback_loop, "_load_canonical_judgment_asset_candidates", lambda *a, **k: [])
    monkeypatch.setattr(feedback_loop, "_load_autoresearch_judgment_asset_candidates", lambda *a, **k: [unrelated])
    monkeypatch.setattr(feedback_loop, "_load_news_judgment_signals", lambda *a, **k: [])

    selected = feedback_loop._select_screening_judgment_asset_candidates(
        industry_major="建設業",
        industry_sub="土木",
        asset_name="油圧ショベル",
        asset_purpose="更新設備の入替",
        hantei="承認",
        limit=3,
    )
    assert selected == []


@pytest.mark.parametrize(
    "case",
    [
        ("製造業", "金属加工", "マシニングセンタ", "生産能力増強", "条件付き承認"),
        ("運送業", "一般貨物", "大型トラック", "配送網の拡大", "承認"),
        ("建設業", "土木", "油圧ショベル", "更新設備の入替", "承認"),
    ],
)
def test_active_canonical_rules_are_offered_for_every_case(case):
    """実データ(data/canonical_judgment_rules.json)の能動ルールが、どの案件でも1件は出せる."""
    rules = feedback_loop._load_canonical_judgment_asset_candidates()
    if not rules:
        pytest.skip("canonical_judgment_rules.json に能動ルールが無い環境")
    industry_major, industry_sub, asset_name, asset_purpose, hantei = case
    terms = feedback_loop._screening_candidate_terms(industry_major, industry_sub, asset_name, asset_purpose, hantei)
    survivors = [
        rule
        for rule in rules
        if feedback_loop._rank_screening_judgment_asset_candidate(rule, terms)[0] > 0
    ]
    assert survivors, f"{case} で提示できる正規判断資産が0件"
