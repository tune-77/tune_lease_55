"""紫苑がユーザーに聞きたいことを記録・一覧する機能（ask_user_question / list_shion_questions）。

既存の紫苑タスク台帳（api.shion_tasks）を source="shion_question" として使うだけの
薄いラッパーであることを検証する。新しいストレージは作らない。
"""
from __future__ import annotations

from api import shion_tasks


def test_ask_user_question_registers_task(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    result = tools.ask_user_question("最近の建設業界の動向を今度教えてください", reason="業種ベンチマークの更新に使いたい")

    assert result["registered"] is True
    assert result["id"]
    assert result["question"] == "最近の建設業界の動向を今度教えてください"

    tasks = shion_tasks.list_tasks(status="open", path=tmp_path / "shion_tasks.jsonl")
    assert len(tasks) == 1
    assert tasks[0]["source"] == "shion_question"
    assert tasks[0]["tags"] == ["question"]
    assert tasks[0]["note"] == "業種ベンチマークの更新に使いたい"


def test_ask_user_question_rejects_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    result = tools.ask_user_question("   ")

    assert result["registered"] is False
    assert not (tmp_path / "shion_tasks.jsonl").exists()


def test_list_shion_questions_filters_by_source(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    tools.ask_user_question("質問A")
    tools.ask_user_question("質問B")
    # 別ソースのタスク（紫苑の質問ではない）は一覧に含めない
    shion_tasks.create_task(title="無関係なタスク", source="manual")

    result = tools.list_shion_questions()

    assert result["count"] == 2
    questions = {q["question"] for q in result["questions"]}
    assert questions == {"質問A", "質問B"}


def test_list_shion_questions_excludes_done(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    result = tools.ask_user_question("回答済みになる質問")
    shion_tasks.set_task_status(result["id"], "done")

    result = tools.list_shion_questions()

    assert result["count"] == 0


def test_execute_tool_dispatches_ask_user_question(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    result = tools.execute_tool("ask_user_question", {"question": "ディスパッチ経由の質問"})
    assert result["registered"] is True

    listed = tools.execute_tool("list_shion_questions", {"limit": 5})
    assert listed["count"] == 1


def test_tool_declarations_include_question_tools():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    names = {item["name"] for item in TOOL_DECLARATIONS}
    assert "ask_user_question" in names
    assert "list_shion_questions" in names
