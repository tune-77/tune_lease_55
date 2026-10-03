from __future__ import annotations

import datetime as dt
import json

from api import chat_prompt_budget as budget


def _bullets(prefix: str, n: int, width: int = 40) -> str:
    return "\n\n【見出し】\n" + "\n".join(f"- {prefix}{i}" + "あ" * width for i in range(n))


def test_core_and_policy_are_never_cut_and_low_tiers_go_first() -> None:
    core = "あなたはリース知性体「紫苑」です。" + "人格" * 400
    blocks = [
        ("base_prompt_root", core),
        ("teaching_prompt_context", _bullets("教示", 5)),
        ("rag_context", _bullets("根拠", 20)),
        ("user_personal_memory_context", _bullets("個人", 60)),
        ("news_brief_context", _bullets("ニュース", 10)),
    ]
    tail = "【社内方針（ユーザーが定めたルール）】\n- 方針: 付き合わない（出典: 判断資産 x）"
    prompt, report = budget.assemble_prompt(blocks, question="根拠 教示", surface="test", max_chars=2500, reserved_tail=tail, log=False)
    assert prompt.startswith(core) and prompt.endswith(tail) and len(prompt) <= 2500
    b = report["blocks"]
    assert b["base_prompt_root"]["kept"] == len(core)
    assert b["teaching_prompt_context"]["kept"] == b["teaching_prompt_context"]["orig"]  # 1層は削られない
    assert b["user_personal_memory_context"]["kept"] < b["user_personal_memory_context"]["orig"]  # 補助から削る
    assert report["dropped_high_priority_chars"] == 0


def test_ordered_blocks_drop_from_the_end_and_history_from_the_start() -> None:
    rag = _bullets("根拠", 200)
    hist = "\n\n【会話履歴の要約】\n" + "\n".join(f"- 発言{i}" + "い" * 30 for i in range(100))
    prompt, report = budget.assemble_prompt([("rag_context", rag), ("chat_history_summary_context", hist)], question="x", surface="test", max_chars=9000, log=False)
    assert "根拠0" in prompt and "根拠199" not in prompt  # 上限を超えたので根拠も予算へ。関連度順なので末尾から
    assert report["blocks"]["rag_context"]["budget_cut"] > 0
    assert report["blocks"]["chat_history_summary_context"]["kept"] == 0  # 根拠より低い層の履歴が先に削られる
    hist_only, _ = budget.assemble_prompt([("chat_history_summary_context", hist)], question="x", surface="test", max_chars=100000, log=False)
    assert "発言99" in hist_only and "発言0い" not in hist_only  # 会話履歴は予算内へ、古いものから
    roomy, report = budget.assemble_prompt([("rag_context", rag)], question="x", surface="test", max_chars=100000, log=False)
    assert roomy == rag and report["blocks"]["rag_context"]["budget_cut"] == 0  # 余裕があれば根拠は削らない


def test_relevance_mode_keeps_items_related_to_the_question() -> None:
    personal = "\n\n【ユーザー個人記憶】\n" + "\n".join(
        [f"- 趣味の話{i}" + "う" * 80 for i in range(60)] + ["- 中古トラックの審査では走行距離を重視する人" + "え" * 40]
    )
    prompt, _ = budget.assemble_prompt([("user_personal_memory_context", personal)], question="中古トラックの走行距離は？", surface="test", max_chars=100000, log=False)
    assert "中古トラックの審査では走行距離" in prompt and len(prompt) <= budget.SPECS["user_personal_memory_context"].budget + 10


def test_duplicate_items_are_kept_only_in_the_higher_priority_block() -> None:
    item = "- 銀行と取引のない企業とは付き合わない 資料がすぐ出てこない企業とも付き合わない"
    blocks = [("memory_recall_context", f"\n\n【紫苑の想起メモ】\n{item}（出典: 判断資産 93d1）\n- 別の記憶です" + "お" * 30),
              ("teaching_prompt_context", f"\n\n【回答前に想起した知識】\n{item}")]
    prompt, report = budget.assemble_prompt(blocks, question="銀行", surface="test", max_chars=100000, log=False)
    assert prompt.count("付き合わない 資料") == 1 and report["blocks"]["memory_recall_context"]["dedup"] > 0
    assert report["blocks"]["teaching_prompt_context"]["dedup"] == 0


def test_log_has_only_names_and_sizes_and_morning_line(tmp_path, monkeypatch) -> None:
    log = tmp_path / "budget.jsonl"
    monkeypatch.setattr(budget, "_LOG_PATH", log)
    secret = "株式会社ひみつ商事の質問"
    budget.assemble_prompt([("base_prompt_root", "人格"), ("rag_context", _bullets("根拠", 400))], question=secret, surface="test", max_chars=3000)
    raw = log.read_text(encoding="utf-8")
    assert "ひみつ" not in raw and "根拠1" not in raw and '"rag_context"' in raw
    now = dt.datetime.now().astimezone()
    line = budget.morning_report_line(log, now=now, threshold=1000)
    assert line and line.startswith("- ✂️ チャットのプロンプト削減（直近24h・1回）") and "rag_context" in line
    assert budget.morning_report_line(log, now=now + dt.timedelta(days=2)) is None
