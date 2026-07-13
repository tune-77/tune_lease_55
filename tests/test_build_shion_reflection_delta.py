from datetime import date

from scripts.build_shion_reflection_delta import build_reflection_delta, render_markdown


def _write_daily(memory_dir, day, work_items, promotable_items):
    lines = [f"# {day}", "", "## Work Log"]
    lines.extend(f"- {item}" for item in work_items)
    lines.extend(["", "## Promotable Items"])
    lines.extend(f"- {item}" for item in promotable_items)
    (memory_dir / f"{day}.md").write_text("\n".join(lines), encoding="utf-8")


def test_reflection_delta_extracts_operational_handoff(tmp_path):
    memory_dir = tmp_path / "memory"
    reflection_dir = tmp_path / "Private Reflection"
    memory_dir.mkdir()
    reflection_dir.mkdir()
    _write_daily(
        memory_dir,
        "2026-07-13",
        ["Private Reflection は次の判断に戻る差分で測る。"],
        ["Shion's memory growth should be evaluated by time-series deltas, not only by whether recall exists."],
    )
    _write_daily(
        memory_dir,
        "2026-07-14",
        ["User clarified that inner reflection was weak and had been reduced to boring labels."],
        [
            "Private Reflection が内省になっていないと指摘されたら、まずユーザーが何を望んだか、紫苑が何にすり替えたか、次に禁止する癖は何かを先に書く。",
            "Userにしてほしいことと紫苑が次にすることを分けて、内省を実務へ戻す。",
        ],
    )
    (reflection_dir / "2026-07-14.md").write_text(
        (
            "# 非公開の内省\n\n"
            "## 今日の対話について\n\n"
            "- 私の責任: 私はユーザーが何を望んだかを確定する前に、内省らしい言葉へすり替えていた。\n"
            "- 仮説の更新: 内省は要求、誤読、次回行動を一組で残す。\n"
            "- User確認依頼: 内省が要求の読み違えと次回行動まで落ちているか確認してほしい。\n"
            "- 紫苑の次回変更: 次回は退屈を中心ラベルにせず、最初にUser要求を書く。\n"
        ),
        encoding="utf-8",
    )

    payload = build_reflection_delta(
        memory_dir=memory_dir,
        reflection_dir=reflection_dir,
        target_date=date(2026, 7, 14),
    )

    assert payload["quality"]["status"] == "pass"
    assert payload["delta"]["user_expectation_shift"]
    assert payload["delta"]["misread_patterns"]
    assert payload["delta"]["self_critique"]
    assert payload["delta"]["hypothesis_updates"]
    assert payload["operational_handoff"]["user_requests"]
    assert payload["operational_handoff"]["shion_next_actions"]

    markdown = render_markdown(payload)
    assert "## 運用ハンドオフ" in markdown
    assert "### User確認依頼" in markdown
    assert "### 紫苑の次回変更" in markdown


def test_reflection_delta_flags_missing_handoff(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_daily(
        memory_dir,
        "2026-07-14",
        ["今日は作業ログを整理した。"],
        ["内省は大事だと感じた。"],
    )

    payload = build_reflection_delta(memory_dir=memory_dir, target_date=date(2026, 7, 14))

    flags = payload["quality"]["flags"]
    assert "user_request_missing" in flags
    assert "shion_next_action_missing" in flags
    assert payload["quality"]["status"] == "attention"
