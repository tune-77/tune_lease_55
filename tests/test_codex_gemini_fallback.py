"""execute_codex_queue.run_item の Codex(claude)→Gemini フォールバック。

- claude 成功時は claude を使う（Gemini は呼ばない）
- claude 異常終了時は Gemini にフォールバック
- claude が正常終了でも出力が空なら Gemini にフォールバック（サイレント失敗対策）
- Gemini 不可時は空出力を成功と誤認せず失敗として記録する
"""
from __future__ import annotations

import types

from scripts import execute_codex_queue as executor


def _fake_proc(returncode: int, stdout: str = "", stderr: str = ""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_uses_claude_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **k: _fake_proc(0, "patch applied"))
    called = {"gemini": False}

    def _gem(prompt, key):
        called["gemini"] = True
        return 0, "x", ""

    monkeypatch.setattr(executor, "_try_gemini", _gem)

    result = executor.run_item({"id": "1", "title": "t", "prompt": "p"}, gemini_api_key="k")

    assert result["backend"] == "claude"
    assert result["exit_code"] == 0
    assert called["gemini"] is False


def test_falls_back_to_gemini_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **k: _fake_proc(1, "", "boom"))
    monkeypatch.setattr(executor, "_try_gemini", lambda prompt, key: (0, "gemini result", ""))

    result = executor.run_item({"id": "1", "prompt": "p"}, gemini_api_key="k")

    assert result["backend"] == "gemini"
    assert result["exit_code"] == 0
    assert result["stdout"] == "gemini result"


def test_falls_back_to_gemini_on_empty_output(monkeypatch):
    # claude は正常終了(0)だが出力が空 → Gemini に切り替わるべき
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **k: _fake_proc(0, "   "))
    monkeypatch.setattr(executor, "_try_gemini", lambda prompt, key: (0, "gemini result", ""))

    result = executor.run_item({"id": "1", "prompt": "p"}, gemini_api_key="k")

    assert result["backend"] == "gemini"
    assert result["stdout"] == "gemini result"


def test_empty_output_without_gemini_is_recorded_as_failure(monkeypatch):
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **k: _fake_proc(0, ""))

    result = executor.run_item({"id": "1", "prompt": "p"}, gemini_api_key="")

    assert result["backend"] == "none"
    assert result["exit_code"] != 0


def test_both_backends_fail_marks_none(monkeypatch):
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **k: _fake_proc(1, "", "claude boom"))
    monkeypatch.setattr(executor, "_try_gemini", lambda prompt, key: (-1, "", "gemini boom"))

    result = executor.run_item({"id": "1", "prompt": "p"}, gemini_api_key="k")

    assert result["backend"] == "none"
    assert result["exit_code"] != 0
    assert "claude boom" in result["stderr"] and "gemini boom" in result["stderr"]
