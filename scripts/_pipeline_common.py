"""Shared helper for reporting a pipeline script's failure reason to stderr.

scripts/detect_pipeline_failures.py の parse_pipeline_log() は
"エラー:"/"ERROR:"/"警告:"/"WARNING:" 接頭辞の行だけを critical_errors/warnings
として拾う。各スクリプトが SystemExit(message) や print() を個別に書いていると
接頭辞の付け忘れで、原因がログに残っていても分類・検出されない（REV-395a/396a/397a
調査時に確認）。ここに一本化する。
"""
from __future__ import annotations

import sys
from pathlib import Path


def report_pipeline_failure(message: str, *, level: str = "エラー") -> None:
    print(f"{level}: {message}", file=sys.stderr)


def write_json_only_stub(
    output_md: Path,
    *,
    title: str,
    json_path: Path,
    generated_at: str = "",
    human_view: str = "reports/shion_memory_sentinel_latest.md",
) -> None:
    """--json-only 時に、人間向けMarkdownを短いスタブで上書きする。

    Markdownを書かずに放置すると、生成をやめた時点の内容が reports/*_latest.md に
    残り続ける。scripts/ops_friction_doctor.py は `reports/*_latest.md` を glob して
    FRICTION_RULES のキーワードで未解決課題を数えるため、凍結したレポートの課題が
    毎朝「現在の課題」として再掲される。スタブで上書きすればヒット0になり、
    人間には統合先のポインタが残る。

    ⚠️ このスタブ本文に FRICTION_RULES のキーワードを入れてはいけない。
    特に memory_pipeline_review ルールの needs_feedback / needs_review /
    human approval / review_required / promotion_queue は、1語でも含めると
    スタブ自身が新しい課題として計上される。
    """
    output_md.parent.mkdir(parents=True, exist_ok=True)
    try:
        json_label = json_path.relative_to(Path(__file__).resolve().parents[1])
    except ValueError:
        json_label = json_path
    stamp = f"（生成: {generated_at}）" if generated_at else ""
    output_md.write_text(
        f"# {title}\n\n"
        f"このMarkdownは --json-only モードのため本文を生成していません{stamp}。\n\n"
        f"- 全データ: `{json_label}`\n"
        f"- 人間向けの統合ビュー: `{human_view}`\n",
        encoding="utf-8",
    )
