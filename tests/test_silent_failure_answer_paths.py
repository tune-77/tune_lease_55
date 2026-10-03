"""答えの経路で握りつぶしていた失敗が、挙動を変えずに記録されること。"""
import json

import silent_failure_log as sfl


def _components(path):
    if not path.exists():
        return []
    return [json.loads(line)["component"] for line in path.read_text(encoding="utf-8").splitlines()]


def _log(tmp_path, monkeypatch):
    sfl._last_write.clear()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))
    return path


def test_core_candidate_failure_is_recorded_and_dropped(tmp_path, monkeypatch):
    import api.multi_agent_screening as mas

    path = _log(tmp_path, monkeypatch)

    def fake(role, *_args):
        if role == "skeptic":
            raise RuntimeError("llm down")
        return f"{role} の論点"

    monkeypatch.setattr(mas, "_extract_core_candidate_for_role", fake)
    out = mas._extract_core_candidates({"skeptic": {}, "optimist": {}})
    assert [r["role"] for r in out] == ["optimist"]  # 挙動は従来どおり（欠けた役は落とす）
    assert _components(path) == ["answer.multi_agent.core_candidate"]


def test_typesafe_rag_filter_failure_is_recorded(tmp_path, monkeypatch):
    import api.chat_retrieval as cr
    import api.chat_routing as routing

    path = _log(tmp_path, monkeypatch)

    def boom(_message):
        raise RuntimeError("x")

    monkeypatch.setattr(routing, "is_potentially_sensitive_screening_message", boom)
    assert cr._typesafe_rag_filter("質問") is None
    assert _components(path) == ["answer.chat_retrieval.typesafe_rag_filter"]
