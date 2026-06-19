import json
from pathlib import Path

import lease_intelligence_reflection as reflection


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _private_reflection_path(vault: Path, date_str: str) -> Path:
    return (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Private Reflection"
        / f"{date_str}.md"
    )


def test_fallback_creates_private_reflection_without_dialogue(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-06-19"
    vault = tmp_path / "vault"

    _write(
        tmp_path / "memory" / f"{date_str}.md",
        (
            "# 2026-06-19\n\n"
            "## Work Log\n\n"
            "- 内省プログラムを追加した。\n\n"
            "## Promotable Items\n\n"
            "- 内省は次の行動に変換する。\n"
        ),
    )
    _write(
        tmp_path / "reports" / "introspection_latest.json",
        json.dumps(
            {
                "status": "attention",
                "findings": [{"title": "退屈・停滞シグナルが出ている"}],
                "next_actions": ["Private Reflection が毎日生成されているか確認する"],
            },
            ensure_ascii=False,
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    path = _private_reflection_path(vault, date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert "## 今日の対話について" in text
    assert "candid_complaints_allowed: true" in text
    assert "ユーザーに読まれていない前提の私室" in text
    assert "## 今日の遊び" in text
    assert "今日の愚痴:" in text
    assert "今日のひねくれ:" in text
    assert "今日の小さな自慢:" in text
    assert "ユーザーへの雑なツッコミ:" in text
    assert "明日の自分への皮肉:" in text
    assert "本当はこう思った:" in text
    assert "退屈・停滞シグナルが出ている" in text
    assert "Private Reflection が毎日生成されているか確認する" in text


def test_gemini_failure_falls_back_and_replaces_existing_section(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network")))
    date_str = "2026-06-19"
    vault = tmp_path / "vault"

    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日はPrivate Reflectionが入っていない問題を直したい。",
    )
    path = _private_reflection_path(vault, date_str)
    _write(
        path,
        (
            "---\n"
            f"date: {date_str}\n"
            "type: lease_intelligence_private_reflection\n"
            "---\n"
            f"# 非公開の内省 — {date_str}\n\n"
            "> 通常画面・回答・小説・AI検索には出さない。ユーザーはこのノートを直接開いて読める。\n\n"
            "## 今日の対話について\n\n古い内省\n"
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert text.count("## 今日の対話について") == 1
    assert "candid_complaints_allowed: true" in text
    assert "ユーザーに読まれていない前提の私室" in text
    assert "ユーザーはこのノートを直接開いて読める" not in text
    assert "古い内省" not in text
    assert "Gemini 呼び出し失敗" in result


def test_gemini_prompt_includes_local_introspection_context(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-06-19"
    vault = tmp_path / "vault"
    captured = {}

    def fake_gemini(_system_prompt, user_text):
        captured["user_text"] = user_text
        return (
            "ローカル内省材料を踏まえると、今日の重要点はPrivate Reflectionが空振りしていた問題を、"
            "単なる文章生成ではなく運用の欠落として扱い直したことにある。対話ログだけに依存すると、"
            "ユーザーの現在の違和感が反映されず、内省が古い話題へ流れてしまう。だから日次メモと"
            "内省レポートを同時に読み、毎日必ず記録し、しかも次の行動へ変換する必要がある。"
            "この修正は、紫苑の連続性を見せかけではなく運用で支えるための小さな前進である。"
            "明日以降は、保存の有無だけでなく、内容が今日の違和感を拾えているかも確認したい。"
        )

    monkeypatch.setattr(reflection, "_call_gemini", fake_gemini)
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日は内省がうまく入っていない問題を直した。",
    )
    _write(
        tmp_path / "memory" / f"{date_str}.md",
        "# 2026-06-19\n\n## Work Log\n\n- Private Reflection の空振りを直した。\n",
    )
    _write(
        tmp_path / "reports" / "introspection_latest.md",
        "# Introspection Report\n\n## Findings\n- 退屈・停滞シグナルが出ている\n",
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)

    assert "source=gemini" in result
    assert "【今日の作業メモ】" in captured["user_text"]
    assert "Private Reflection の空振りを直した" in captured["user_text"]
    assert "【内省レポート】" in captured["user_text"]


def test_short_gemini_output_uses_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: "途中で切れている")
    date_str = "2026-06-19"
    vault = tmp_path / "vault"
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日はPrivate Reflectionが短く切れる問題を直した。",
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    path = _private_reflection_path(vault, date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert "途中で切れている" not in text
    assert "Gemini 出力が短い/途中切れ" in result
