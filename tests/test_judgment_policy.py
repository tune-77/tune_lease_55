from __future__ import annotations

import json

import pytest

from api import chat_teaching_capture as teaching
from api.judgment_policy import POLICY_HEADER, classify_knowledge_kind, knowledge_kind_of
from scripts import backfill_knowledge_kind as backfill


@pytest.mark.parametrize(
    "text",
    [
        "銀行と取引のない企業とは付き合わない 資料がすぐ出てこない企業とも付き合わない",
        "メイン銀行の借り入れよりもリースの借り入れが増えないようにする",
        "延滞がちな企業にはリース増額しない",
        "個人保証なしの案件は取り扱わない",
        "決算書は必ず3期分をもらう",
        "当社の方針として反社チェック未了の先とは契約しない",
    ],
)
def test_explicit_company_rules_are_policy(text) -> None:
    assert classify_knowledge_kind(text) == "policy"


@pytest.mark.parametrize(
    "text",
    [
        "3月と9月にサプライヤーの売り込みがありリース契約が増加する傾向がある",
        "新車登録が5年以内で走行距離が200,000キロ位だったらばリースでもやっちゃう",
        "中古自動車の見積書は要注意。合計欄の金額にリサイクル料金が別途かかる場合がある",
        "オンバランス化は必ずしも実質的な財務悪化を意味しない",
        "金融機関の大断る必要がある場合がある",
        "汎用性の高い物件は陳腐化しないので残価が安定する",
        "この物件は中古市場で取り扱われることが多い",
    ],
)
def test_tendencies_numbers_procedures_and_hedges_are_insight(text) -> None:
    assert classify_knowledge_kind(text) == "insight"


def test_stored_kind_wins_over_reclassification() -> None:
    assert knowledge_kind_of({"knowledge_kind": "insight", "claim": "取り扱わない"}) == "insight"
    assert knowledge_kind_of({"claim": "取り扱わない"}) == "policy"


def test_teaching_recall_block_puts_policy_first_with_citation() -> None:
    items = [
        {"topic": "2026-10-02に教わった判断", "snippet": "銀行と取引のない企業とは付き合わない", "user_taught": True,
         "citation": "判断資産候補 abc123・2026-10-02 教示"},
        {"topic": "2026-09-30に教わった判断", "snippet": "走行距離20万キロ・5年以内ならリース可", "user_taught": True,
         "citation": "判断資産候補 def456・2026-09-30 教示"},
    ]
    block = teaching.build_recall_prompt_block(items)
    assert block.startswith(POLICY_HEADER)
    assert "冒頭で結論として述べる" in block and "方針を優先し" in block
    policy_part, insight_part = block.split("【回答前に想起した知識】")
    assert "付き合わない（出典: 判断資産候補 abc123・2026-10-02 教示）" in policy_part
    assert "付き合わない" not in insight_part and "走行距離20万キロ" in insight_part and "def456" in insight_part
    assert "出典" in insight_part


def test_memory_recall_block_lifts_canonical_policies(monkeypatch) -> None:
    from api import shion_memory_recall as recall

    memories = [
        {"source": "canonical_judgment_rules", "content": "メイン銀行の借り入れよりもリースの借り入れが増えないようにする", "judgment_asset_id": "8932df45", "created_at": "2026-10-02", "memory_type": "judgment_memory"},
        {"source": "canonical_judgment_rules", "content": "工作機械は汎用性が高く中古流動性がある", "judgment_asset_id": "aaa111", "created_at": "2026-09-20", "memory_type": "judgment_memory"},
        {"source": "long_term_memory", "content": "取り扱わない（MEMORY.md の古い記述）", "memory_type": "judgment_memory"},
    ]
    monkeypatch.setattr(recall, "recall_memories", lambda q, limit, index_path=None: {"memories": memories, "route": "case_screening", "refs": []})
    block, _ = recall.build_recall_prompt_block("メイン銀行とリース残高は？", log_usage=False)
    head, rest = block.split("【紫苑の想起メモ】")
    assert head.startswith(POLICY_HEADER) and "増えないようにする（出典: 判断資産 8932df45・2026-10-02 教示）" in head
    assert "工作機械は汎用性が高く中古流動性がある（出典: 判断資産 aaa111・2026-09-20 教示）" in rest
    assert "MEMORY.md の古い記述" in rest  # 判断資産以外は方針扱いしない


def test_backfill_adds_kind_once(tmp_path) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{"canonical_statement": "銀行と取引のない企業とは付き合わない"}, {"canonical_statement": "3月に増える傾向", "knowledge_kind": "insight"}]}, ensure_ascii=False))
    cands = tmp_path / "c.jsonl"
    cands.write_text(json.dumps({"claim": "必ず3期分の決算書をもらう"}, ensure_ascii=False) + "\n")
    assert backfill.backfill_rules(rules, dry_run=False) == {"added": 1, "policy": 1}
    assert backfill.backfill_candidates(cands, dry_run=False) == {"added": 1, "policy": 1}
    assert backfill.backfill_rules(rules, dry_run=False) == {"added": 0, "policy": 0}
    assert json.loads(rules.read_text())["rules"][0]["knowledge_kind"] == "policy"
