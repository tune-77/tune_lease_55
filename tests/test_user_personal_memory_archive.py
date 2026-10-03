import datetime as dt
import json

from api import user_personal_memory_archive as upa
from api.chat_user_personal_memory import invalidate_user_personal_memory_cache, load_user_personal_memory_payload

TODAY = dt.date(2026, 10, 3)


def _setup(tmp_path, monkeypatch, memory_md: str, personal_md: str = "", rules: list | None = None):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUDRUN_PENDING_GCS_ENABLED", raising=False)
    invalidate_user_personal_memory_cache()
    data = tmp_path / "data"
    data.mkdir()
    (data / "user_personal_memory.md").write_text(personal_md, encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text(memory_md, encoding="utf-8")
    (data / "canonical_judgment_rules.json").write_text(json.dumps({"rules": rules or []}, ensure_ascii=False), encoding="utf-8")
    return lambda name: str(data / name)


def _block(tmp_path, resolver) -> str:
    invalidate_user_personal_memory_cache()
    return load_user_personal_memory_payload(repo_root=tmp_path, data_path_resolver=resolver)["block"]


def test_weekly_run_archives_obvious_items_and_keeps_relationship(tmp_path, monkeypatch):
    memory_md = "\n".join([
        "## Auto Promotions 2026-06-01 04:00",
        "- [2026-06-01] 紫苑のぼやきは `memory/x.md` を参照して硬くしない方針。",
        "- [2026-06-01] 紫苑はUserの呼び方を大事にする。",
        "## Preferences",
        "- 紫苑の方針: 結論を先に短く返す。",
        "- 紫苑の方針: 結論を先に短く返す 。",
    ])
    personal_md = "\n".join([
        "## Personal Facts",
        "- [confirmed] Dog name: タム",
        "- 2026-07-03T19:55:35 [confirmed/family_pet] (chat) 僕の犬の名前は何だっけ？",
        "- 2026-07-03T12:05:12 [confirmed/explicit_remember] (chat) ホテル 空調機はリースを使うことが多い 覚えておいて",
    ])
    resolver = _setup(tmp_path, monkeypatch, memory_md, personal_md, [{"status": "active", "canonical_statement": "ホテル 空調機はリースを使うことが多い"}])

    report = upa.run(dry_run=False, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY)

    reasons = sorted(a["reason"] for a in report["auto"])
    assert reasons == ["covered_by_judgment_asset", "duplicate", "question_not_fact", "stale_90d"]
    assert report["review_candidates"] == 1  # 古いが関係性の語（呼び方・大事）を含む行は人の判断へ
    block = _block(tmp_path, resolver)
    assert "Dog name: タム" in block and "呼び方を大事に" in block
    assert "何だっけ" not in block and "空調機" not in block and "memory/x.md" not in block
    assert block.count("結論を先に短く返す") == 1
    assert report["after"]["chars"] < report["before"]["chars"]
    assert "今回アーカイブ 4件" in upa.morning_report_line(tmp_path / "data" / "user_personal_memory_archive.json")


def test_archived_lines_do_not_let_older_lines_slip_in_under_the_limit(tmp_path, monkeypatch):
    memory_md = "\n".join(f"- 紫苑メモ{i}" for i in range(40))
    resolver = _setup(tmp_path, monkeypatch, memory_md)
    archive = tmp_path / "data" / "user_personal_memory_archive.json"
    upa.save_archive({"items": [{"id": "x", "key": upa.line_key(f"- 紫苑メモ{i}")} for i in range(40)], "pinned": []}, archive)

    assert "紫苑メモ" not in _block(tmp_path, resolver)


def test_restore_brings_line_back_and_pins_it(tmp_path, monkeypatch):
    memory_md = "## Auto Promotions 2026-05-01 04:00\n- [2026-05-01] 紫苑は短く答える。"
    resolver = _setup(tmp_path, monkeypatch, memory_md)
    archive = tmp_path / "data" / "user_personal_memory_archive.json"
    report = upa.run(dry_run=False, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY)
    assert "短く答える" not in _block(tmp_path, resolver)

    assert upa.restore(report["auto"][0]["id"], archive_path=archive)["restored"] is True
    assert "短く答える" in _block(tmp_path, resolver)
    again = upa.run(dry_run=False, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY)
    assert again["auto_archived"] == 0  # 戻した行は週次で再アーカイブしない


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    resolver = _setup(tmp_path, monkeypatch, "## Auto Promotions 2026-05-01 04:00\n- [2026-05-01] 紫苑メモ")
    report = upa.run(dry_run=True, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY)
    assert report["auto_archived"] == 1
    assert not (tmp_path / "data" / "user_personal_memory_archive.json").exists()


def test_jev_only_lists_review_candidates_and_never_sends_sensitive_lines(tmp_path, monkeypatch):
    memory_md = "\n".join([
        "## Preferences",
        "- 紫苑の方針A: 結論を先に返す。",
        "- 紫苑の方針B: 結論から先に返す。",
        "- Mana は亡くなった妹さんの名を託した紫苑の方針。",
        "- 紫苑の方針: タムの話は短く。",
    ])
    resolver = _setup(tmp_path, monkeypatch, memory_md, "## Personal Facts\n- [confirmed] Dog name: タム")
    upa.save_archive({"items": [{"id": "d", "key": upa.line_key("- [confirmed] Dog name: タム"), "text": "- [confirmed] Dog name: タム"}], "pinned": []}, tmp_path / "data" / "user_personal_memory_archive.json")
    monkeypatch.setattr("typesafe_dedup_guard.is_safe_public_candidate", lambda c: True)
    sent: list[str] = []

    def scorer(pairs, question):
        sent.extend(a + b for a, b in pairs)
        return [0.9 if "方針A" in a + b and "方針B" in a + b else 0.1 for a, b in pairs]

    report = upa.run(dry_run=False, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY, pair_scorer=scorer)

    assert report["auto_archived"] == 0  # Jev 単独では自動アーカイブしない
    review = upa.load_archive(tmp_path / "data" / "user_personal_memory_archive.json")["review_candidates"]
    assert [r["reason"] for r in review] == ["jev_duplicate"]
    assert sent and not any("妹" in s or "タム" in s for s in sent)


def test_jev_failure_keeps_rule_based_cleanup(tmp_path, monkeypatch):
    resolver = _setup(tmp_path, monkeypatch, "## Auto Promotions 2026-05-01 04:00\n- [2026-05-01] 紫苑メモ\n## Preferences\n- 紫苑の方針: 結論を先に\n- 紫苑の好きな季節は秋")
    monkeypatch.setattr("typesafe_dedup_guard.is_safe_public_candidate", lambda c: True)

    def boom(pairs, question):
        raise RuntimeError("down")

    report = upa.run(dry_run=False, repo_root=tmp_path, data_path_resolver=resolver, today=TODAY, pair_scorer=boom)
    assert report["auto_archived"] == 1 and report["jev"].startswith("skipped")
