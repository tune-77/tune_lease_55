from __future__ import annotations

from scripts import build_wiki_promotion_queue as wiki


def _doc(path: str, content: str = "本文") -> dict:
    return {"path": path, "title": path, "content": content}


def test_build_queue_scores_chat_log_source_docs(monkeypatch):
    docs = [
        _doc("Daily/2026-07-12.md", "判断・承認条件・稟議のメモ。改善点も書く。"),
        _doc("Projects/tune_lease_55/Asset Knowledge/excavator.md", "資産知識ページ。候補源ではない。"),
    ]
    monkeypatch.setattr(wiki, "load_documents", lambda: docs)

    queue = wiki.build_queue(limit=3)

    assert queue["total_docs"] == 2
    assert queue["candidate_source_docs"] == 1
    assert queue["candidate_count"] == 1


def test_main_warns_and_exits_nonzero_when_chat_log_markers_drift(tmp_path, monkeypatch, capsys):
    """ドキュメントは十分読めているのにCHAT_LOG_MARKERSに一致するパスが
    0件の場合、無音でexit 0にしないことを確認する回帰テスト。"""
    docs = [_doc(f"Some/Other/Path/{i}.md") for i in range(10)]
    monkeypatch.setattr(wiki, "load_documents", lambda: docs)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_wiki_promotion_queue.py",
            "--latest", str(tmp_path / "latest.json"),
            "--output", str(tmp_path / "queue.json"),
        ],
    )

    exit_code = wiki.main()

    assert exit_code == 1
    assert "ドリフト" in capsys.readouterr().err


def test_main_returns_zero_when_too_few_docs_to_judge_drift(tmp_path, monkeypatch):
    """読めたドキュメントが閾値未満なら誤検知せず正常終了する。"""
    docs = [_doc("Some/Other/Path/only-one.md")]
    monkeypatch.setattr(wiki, "load_documents", lambda: docs)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_wiki_promotion_queue.py",
            "--latest", str(tmp_path / "latest.json"),
            "--output", str(tmp_path / "queue.json"),
        ],
    )

    assert wiki.main() == 0


def test_main_returns_zero_when_chat_log_docs_found_normally(tmp_path, monkeypatch):
    docs = [_doc(f"Daily/2026-07-{i:02d}.md", "判断・承認条件のメモ") for i in range(1, 12)]
    monkeypatch.setattr(wiki, "load_documents", lambda: docs)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_wiki_promotion_queue.py",
            "--latest", str(tmp_path / "latest.json"),
            "--output", str(tmp_path / "queue.json"),
        ],
    )

    assert wiki.main() == 0
