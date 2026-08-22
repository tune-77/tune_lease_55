import json

from api.chat_prompt_cost import build_composition, estimate_tokens
from scripts import report_chat_prompt_composition as report


def test_estimate_tokens_is_zero_for_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("あいう") == 1


def test_build_composition_records_blocks_and_unaccounted():
    blocks = {
        "rag_context": "あ" * 300,
        "memory_recall_context": "い" * 90,
        "empty_block": "",
    }
    # 名前を付けていないブロックが 60 文字ぶん混ざった状態
    final_prompt = "あ" * 300 + "い" * 90 + "う" * 60

    entry = build_composition(
        surface="next_chat_rag",
        blocks=blocks,
        final_prompt=final_prompt,
        base_prompt=final_prompt,
    )

    assert entry["surface"] == "next_chat_rag"
    # 空ブロックは記録しない
    assert set(entry["blocks"]) == {"rag_context", "memory_recall_context"}
    assert entry["blocks"]["rag_context"]["chars"] == 300
    assert entry["blocks"]["rag_context"]["tokens"] == 100
    assert entry["measured_chars"] == 390
    assert entry["unaccounted_chars"] == 60
    assert entry["block_count"] == 2
    assert entry["capped"] is False
    assert entry["dropped_chars"] == 0


def test_build_composition_flags_capped_prompt():
    base = "あ" * 1000
    entry = build_composition(
        surface="next_chat_general",
        blocks={"base_system_root": base},
        final_prompt="あ" * 600,
        base_prompt=base,
    )

    assert entry["capped"] is True
    assert entry["dropped_chars"] == 400
    # 削られた後の合計より計測値が大きくても unaccounted は負にしない
    assert entry["unaccounted_chars"] == 0


def test_report_ranks_blocks_by_average_tokens(tmp_path):
    path = tmp_path / "chat_prompt_composition.jsonl"
    rows = [
        {
            "surface": "next_chat_rag",
            "final_tokens": 500,
            "unaccounted_chars": 30,
            "capped": False,
            "blocks": {
                "rag_context": {"chars": 900, "tokens": 300},
                "memory_recall_context": {"chars": 90, "tokens": 30},
            },
        },
        {
            "surface": "next_chat_rag",
            "final_tokens": 700,
            "unaccounted_chars": 30,
            "capped": True,
            "blocks": {
                "rag_context": {"chars": 1500, "tokens": 500},
                "memory_recall_context": {"chars": 60, "tokens": 20},
            },
        },
        {
            "surface": "next_chat_general",
            "final_tokens": 100,
            "unaccounted_chars": 0,
            "capped": False,
            "blocks": {"memory_recall_context": {"chars": 30, "tokens": 10}},
        },
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    summary = report.summarize(report.read_entries(path, surface="next_chat_rag"))

    assert summary["samples"] == 2
    assert summary["capped_samples"] == 1
    assert summary["avg_final_tokens"] == 600.0
    # 平均トークンの降順。一番食っているブロックが先頭に来る。
    assert [b["block"] for b in summary["blocks"]] == ["rag_context", "memory_recall_context"]
    assert summary["blocks"][0]["avg_tokens"] == 400.0

    md = report.render(summary)
    assert "rag_context" in md
    assert "Avg final tokens: 600.0" in md


def test_report_handles_missing_log(tmp_path):
    summary = report.summarize(report.read_entries(tmp_path / "nope.jsonl"))

    assert summary["samples"] == 0
    assert summary["blocks"] == []
    assert "計測データなし" in report.render(summary)
