"""紫苑がユーザーに聞きたいことを記録・一覧・回答受領する機能。

既存の紫苑タスク台帳（api.shion_tasks）を source="shion_question" として使うだけの
薄いラッパーであることを検証する。新しいストレージは作らない。
answer_shion_question は「聞く→答えをもらう→（個人的な内容なら）記憶に昇格する」の
輪を閉じる。業務知識としての正式な記憶化（canonical_judgment_rules等）は既存の
人間レビュー付き昇格フロー任せで、ここでは行わないことも検証する。
"""
from __future__ import annotations

from api import shion_tasks, user_personal_memory


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
    assert "answer_shion_question" in names


def test_answer_shion_question_marks_done_and_records_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    monkeypatch.setattr(user_personal_memory, "LOCAL_MEMORY_PATH", tmp_path / "user_personal_memory.md")
    import lease_intelligence_tools as tools

    asked = tools.ask_user_question("最近の建設業界の動向を今度教えてください", reason="ベンチマーク更新用")
    result = tools.answer_shion_question(asked["id"], "堅調に推移しています")

    assert result["recorded"] is True
    assert result["personal_memory_captured"] is False  # 業務的な回答は個人記憶にしない

    tasks = shion_tasks.list_tasks(status="all", path=tmp_path / "shion_tasks.jsonl")
    task = next(t for t in tasks if t["id"] == asked["id"])
    assert task["status"] == "done"
    assert "ベンチマーク更新用" in task["note"]
    assert "堅調に推移しています" in task["note"]

    # 回答済みなので一覧から消える
    assert tools.list_shion_questions()["count"] == 0


def test_answer_shion_question_captures_personal_memory_when_applicable(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    monkeypatch.setattr(user_personal_memory, "LOCAL_MEMORY_PATH", tmp_path / "user_personal_memory.md")
    import lease_intelligence_tools as tools

    asked = tools.ask_user_question("お名前を教えてください")
    result = tools.answer_shion_question(asked["id"], "覚えて。犬の名前はポチです")

    assert result["personal_memory_captured"] is True
    memory_text = (tmp_path / "user_personal_memory.md").read_text(encoding="utf-8")
    assert "ポチ" in memory_text


def test_answer_shion_question_rejects_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    result = tools.answer_shion_question("task_does_not_exist", "回答")

    assert result["recorded"] is False


def test_answer_shion_question_rejects_non_question_task(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    import lease_intelligence_tools as tools

    manual_task = shion_tasks.create_task(title="無関係なタスク", source="manual", path=tmp_path / "shion_tasks.jsonl")

    result = tools.answer_shion_question(manual_task["id"], "回答")

    assert result["recorded"] is False


def test_execute_tool_dispatches_answer_shion_question(tmp_path, monkeypatch):
    monkeypatch.setattr(shion_tasks, "TASK_LOG_PATH", tmp_path / "shion_tasks.jsonl")
    monkeypatch.setattr(user_personal_memory, "LOCAL_MEMORY_PATH", tmp_path / "user_personal_memory.md")
    import lease_intelligence_tools as tools

    asked = tools.execute_tool("ask_user_question", {"question": "質問X"})
    result = tools.execute_tool(
        "answer_shion_question", {"question_id": asked["id"], "answer": "回答X"}
    )

    assert result["recorded"] is True
