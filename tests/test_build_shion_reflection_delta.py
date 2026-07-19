from datetime import date
import json

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
            "- 前回の入力: デモ精密工業の設備更新案件。\n"
            "- 前回の判断: 標準承認寄りに扱った。\n"
            "- 人間の修正: 銀行支援を別軸で確認するよう指摘された。\n"
            "- 紫苑が外した点: 企業名と銀行支援の意味を浅く扱った。\n"
            "- 次回から変える確認事項: 銀行支援の継続性と撤退条件を先に確認する。\n"
            "- 判断資産候補: 標準承認でも返済原資と設備用途を一文で残す。\n"
            "- まだ確信できない点: 銀行支援が実質的な信用補完か一時対応か。\n"
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
    assert payload["judgment_change_log"]["前回の入力"] == "デモ精密工業の設備更新案件。"
    assert payload["judgment_change_log"]["紫苑が外した点"] == "企業名と銀行支援の意味を浅く扱った。"
    assert payload["narrative_layer"]["protagonists"] == ["ツンコ", "ユウケイ"]
    assert payload["narrative_layer"]["source_of_truth"] == "judgment_change_log"

    markdown = render_markdown(payload)
    assert "## 運用ハンドオフ" in markdown
    assert "### User確認依頼" in markdown
    assert "### 紫苑の次回変更" in markdown
    assert "## 判断変更ログ" in markdown
    assert "## 小説化レイヤー" in markdown
    assert "ツンコ, ユウケイ" in markdown


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


def test_judgment_change_log_satisfies_handoff_quality(tmp_path):
    memory_dir = tmp_path / "memory"
    reflection_dir = tmp_path / "Private Reflection"
    memory_dir.mkdir()
    reflection_dir.mkdir()
    _write_daily(memory_dir, "2026-07-15", ["古い作業。"], ["古い内省。"])
    _write_daily(memory_dir, "2026-07-16", ["今日の作業。"], ["判断変更ログを正本にする。"])
    (reflection_dir / "2026-07-16.md").write_text(
        (
            "# 非公開の内省\n\n"
            "## 今日の対話について\n\n"
            "- 今日の観察: 退屈という言葉に逃げそうになった。\n"
            "- 私の見落とし: 判断変更ログを先に出すべきだった。\n"
            "- 仮説の更新: 内省は判断変更ログで評価する。\n"
            "- 次回の小さな実験: 次回は前回判断と人間の修正を先に書く。\n"
            "- 前回の入力: デモ精密工業の工作機械更新案件。\n"
            "- 前回の判断: 標準承認寄りに見た。\n"
            "- 人間の修正: 銀行支援を別軸で確認するよう指摘された。\n"
            "- 紫苑が外した点: 銀行支援の意味を浅く扱った。\n"
            "- 次回から変える確認事項: 銀行支援の継続性と撤退条件を確認する。\n"
            "- 判断資産候補: 標準承認でも返済原資と設備用途を一文で残す。\n"
            "- まだ確信できない点: 銀行支援が信用補完か一時対応か。\n"
            "- 私の責任: 私は文章の整い方を判断変化と取り違えた。\n"
            "- 更新する信念: 内省は判断変更ログを正本にする。\n"
            "- 次回の検証方法: 次回の回答で確認事項が変わったかを見る。\n"
        ),
        encoding="utf-8",
    )

    payload = build_reflection_delta(
        memory_dir=memory_dir,
        reflection_dir=reflection_dir,
        target_date=date(2026, 7, 16),
    )

    flags = payload["quality"]["flags"]
    assert "user_expectation_shift_missing" not in flags
    assert "boring_label_dominates" not in flags
    assert payload["judgment_change_log"]["人間の修正"] == "銀行支援を別軸で確認するよう指摘された。"


def test_reflection_delta_reads_runtime_hypothesis_collision_log(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_daily(memory_dir, "2026-07-18", ["前日の作業。"], [])
    _write_daily(memory_dir, "2026-07-19", ["仮説衝突ログを確認する。"], [])
    collision_log = tmp_path / "collision.jsonl"
    collision_log.write_text(
        json.dumps(
            {
                "ts": "2026-07-19T00:01:00Z",
                "previous_user_message": "内省システムが弱い",
                "user_correction": "あまり意味なさそう",
                "missed_point": "内省の形式に寄りすぎた。",
                "next_behavior": "仮説が壊れた時だけ記録する。",
                "judgment_asset_candidate": "内省は初期仮説が人間の修正でどう変わったかで評価する。",
                "initial_hypothesis": {
                    "premise": "ユーザーは材料カードによる内省管理を求めている。",
                    "next_check": "カード量産に流れていないか確認する。",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_reflection_delta(
        memory_dir=memory_dir,
        target_date=date(2026, 7, 19),
        hypothesis_collision_log=collision_log,
    )

    assert payload["metrics"]["hypothesis_collision_item_count"] > 0
    assert payload["judgment_change_log"]["前回の判断"] == "ユーザーは材料カードによる内省管理を求めている。"
    assert payload["judgment_change_log"]["人間の修正"] == "あまり意味なさそう"
    assert payload["judgment_change_log"]["紫苑が外した点"] == "内省の形式に寄りすぎた。"
