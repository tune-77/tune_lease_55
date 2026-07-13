from datetime import date

from scripts.build_shion_timeline_delta import build_timeline_delta, render_markdown


def _write_daily(memory_dir, day, work_items, promotable_items):
    lines = [f"# {day}", "", "## Work Log"]
    lines.extend(f"- {item}" for item in work_items)
    lines.extend(["", "## Promotable Items"])
    lines.extend(f"- {item}" for item in promotable_items)
    (memory_dir / f"{day}.md").write_text("\n".join(lines), encoding="utf-8")


def test_build_timeline_delta_detects_continued_new_and_dropped_terms(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_daily(
        memory_dir,
        "2026-07-10",
        ["紫苑の記憶活用は時系列で比較し、前回との差分を確認する。"],
        ["次回は同種案件で質問の聞き方を検証する。"],
    )
    _write_daily(
        memory_dir,
        "2026-07-11",
        ["Private Reflection は文章量ではなく、次の判断に戻る差分で測る。"],
        ["記憶活用では前回からの変化を短く表示する。"],
    )
    _write_daily(
        memory_dir,
        "2026-07-12",
        ["紫苑は判断資産エージェントとして面白いが、運用はまだ未検証。"],
        ["次回はユーザーの圧点を確認し、余計な質問を避ける。"],
    )
    _write_daily(
        memory_dir,
        "2026-07-13",
        ["紫苑の記憶活用はまだ弱いので、1日前2日前3日前の時系列差分を試す。"],
        ["次回は同種案件で前回より聞き方と判断理由が良くなったか比較する。"],
    )

    payload = build_timeline_delta(memory_dir, date(2026, 7, 13), days=4)

    delta = payload["delta"]
    assert "前回" in delta["continued_terms"]
    assert "判断理由" in delta["new_terms"]
    assert "圧点" in delta["dropped_terms"]
    assert payload["interpretation"]["user_pressure_points"]
    assert payload["interpretation"]["next_behavior_candidates"]

    layers = payload["memory_layers"]
    assert layers["short_term"]["items"]
    assert "前回" in layers["mid_term"]["signals"]["repeated_terms"]
    assert layers["long_term"]["promotion_candidates"]
    assert "短期は会話運び" in layers["anti_random_rule"]


def test_render_markdown_handles_empty_sections(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_daily(memory_dir, "2026-07-13", ["短い作業だけを記録する。"], [])

    payload = build_timeline_delta(memory_dir, date(2026, 7, 13), days=2)
    markdown = render_markdown(payload)

    assert "# Shion Timeline Delta - 2026-07-13" in markdown
    assert "## 差分サマリ" in markdown
    assert "### 継続している論点" in markdown
    assert "## 記憶レイヤー" in markdown
