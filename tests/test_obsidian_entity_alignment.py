from __future__ import annotations

from pathlib import Path

import obsidian_entity_alignment as alignment


def _write_note(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_notes_only_reads_safe_included_roots(tmp_path: Path) -> None:
    _write_note(
        tmp_path / "03-知識_業界" / "残価.md",
        "---\naliases: [中古価値]\ntags: [審査, 物件]\n---\n# 残価\n満了時の価値。",
    )
    _write_note(tmp_path / "Daily" / "2026-09-20.md", "# 日報\n秘密")
    _write_note(tmp_path / "03-知識_業界" / "screening_records.md", "# 案件一覧")

    notes = alignment.load_notes(tmp_path)

    assert len(notes) == 1
    assert notes[0].title == "残価"
    assert notes[0].aliases == ("中古価値",)
    assert notes[0].tags == ("審査", "物件")
    assert notes[0].path == "03-知識_業界/残価.md"


def test_load_notes_excludes_sensitive_nested_directories(tmp_path: Path) -> None:
    _write_note(tmp_path / "03-知識_業界" / "公開.md", "# 公開\n一般知識")
    _write_note(
        tmp_path / "03-知識_業界" / "past_cases" / "customer-123.md",
        "# 顧客案件\n非公開情報",
    )

    notes = alignment.load_notes(tmp_path)

    assert [note.title for note in notes] == ["公開"]


def test_outline_skips_headings_inside_code_fences() -> None:
    body = """# 公開見出し
```markdown
## 案件固有の秘密
```
~~~text
### 顧客識別子
~~~
## 公開サブ見出し
"""

    assert alignment._outline(body) == ("公開見出し", "公開サブ見出し")


def test_load_notes_skips_a_note_that_times_out(tmp_path: Path, monkeypatch) -> None:
    _write_note(tmp_path / "03-知識_業界" / "読める.md", "# 読める\n本文")
    _write_note(tmp_path / "03-知識_業界" / "遅い.md", "# 遅い\n本文")
    original = alignment._read_text_with_timeout

    def fake_read(path: Path, timeout_seconds: float) -> str:
        if path.name == "遅い.md":
            raise TimeoutError("iCloud placeholder")
        return original(path, timeout_seconds)

    monkeypatch.setattr(alignment, "_read_text_with_timeout", fake_read)

    notes = alignment.load_notes(tmp_path)

    assert [note.title for note in notes] == ["読める"]


def test_dataless_flag_detection_uses_macos_placeholder_bit(monkeypatch) -> None:
    class _Stat:
        st_flags = alignment.MACOS_DATALESS_FLAG

    path = Path("placeholder.md")
    monkeypatch.setattr(Path, "stat", lambda _self: _Stat())

    assert alignment._is_dataless(path) is True


def test_candidate_selection_uses_aliases_and_existing_links() -> None:
    notes = [
        alignment.NoteEntity("a.md", "残価評価", ("中古価値",), ("物件",), "説明A", ("中古価値",)),
        alignment.NoteEntity("b.md", "中古価値", (), ("物件",), "説明B", ()),
        alignment.NoteEntity("c.md", "為替リスク", (), ("市場",), "説明C", ()),
    ]

    pairs = alignment.select_candidate_pairs(notes, max_pairs=10)

    assert [(pair.a, pair.b) for pair in pairs] == [(0, 1)]
    assert "title_or_alias_match" in pairs[0].reasons
    assert pairs[0].already_linked is True


def test_existing_link_detection_accepts_full_path_and_filename_stem() -> None:
    notes = [
        alignment.NoteEntity(
            "03-知識_業界/市場トレンド_2026.md",
            "リース市場の今後見込み 2026年版",
            (),
            (),
            "",
            ("補助金×ESGリース2026",),
        ),
        alignment.NoteEntity(
            "03-知識_業界/補助金・融資/補助金×ESGリース2026.md",
            "補助金×ESGリース 2026年版",
            (),
            (),
            "",
            (),
        ),
    ]

    pairs = alignment.select_candidate_pairs(notes, min_similarity=0.1, max_pairs=10)

    assert len(pairs) == 1
    assert pairs[0].already_linked is True


def test_candidate_selection_ignores_domain_wide_title_fragments() -> None:
    notes = [
        alignment.NoteEntity("a.md", "リース基本ルール autoreserch", (), (), "", ()),
        alignment.NoteEntity("b.md", "再リース autoreserch", (), (), "", ()),
        alignment.NoteEntity("c.md", "メンテナンスリース autoreserch", (), (), "", ()),
        alignment.NoteEntity("d.md", "リース動産保険 autoreserch", (), (), "", ()),
    ]

    pairs = alignment.select_candidate_pairs(notes, min_similarity=0.3, max_pairs=20)

    assert pairs == []


def test_request_omits_local_paths_and_has_score_plus_nouls() -> None:
    notes = [
        alignment.NoteEntity("private/a.md", "残価", (), ("審査",), "説明A", ()),
        alignment.NoteEntity("private/b.md", "中古価値", (), ("審査",), "説明B", ()),
    ]
    pair = alignment.CandidatePair(0, 1, 0.5, ("title_similarity",), False)

    payload = alignment.build_alignment_request(notes, [pair])

    assert "private/a.md" not in str(payload)
    assert payload["questions"]["pair0_alignment"]["type"] == "score"
    assert payload["questions"]["pair0_same_subject"]["type"] == "noul"
    assert payload["questions"]["pair0_same_conclusion"]["type"] == "noul"
    assert payload["questions"]["pair0_contradiction"]["type"] == "noul"
    assert "excerpt" not in payload["state"]["pairs"][0]["note_a"]


def test_judge_pairs_routes_related_candidate() -> None:
    notes = [
        alignment.NoteEntity("a.md", "残価", (), (), "満了時価値", ()),
        alignment.NoteEntity("b.md", "中古市場", (), (), "売却市場", ()),
    ]
    pairs = [alignment.CandidatePair(0, 1, 0.4, ("title_similarity",), False)]

    def fake_request(_payload):
        return {
            "model": "jev-test",
            "answers": {
                "pair0_alignment": {
                    "type": "score",
                    "score": 1.1,
                    "confidence": 0.8,
                    "probabilities": {"0": 0.0, "1": 0.9, "2": 0.1},
                },
                "pair0_same_subject": {"type": "noul", "noul": 0.7},
                "pair0_same_conclusion": {"type": "noul", "noul": 0.4},
                "pair0_contradiction": {"type": "noul", "noul": 0.1},
            },
            "usage": {"input_tokens": 200},
        }

    judged, meta = alignment.judge_pairs(notes, pairs, request_fn=fake_request)

    assert judged[0]["route"] == "related_link_candidate"
    assert meta["status"] == "applied"
    assert meta["usage"] == {"input_tokens": 200}


def test_contradiction_and_low_confidence_override_score_route() -> None:
    assert (
        alignment.route_alignment(score=1.9, confidence=0.95, contradiction=0.9, already_linked=False)
        == "contradiction_review"
    )
    assert (
        alignment.route_alignment(score=1.9, confidence=0.2, contradiction=0.1, already_linked=False)
        == "uncertain_review"
    )


def test_live_feature_requires_its_own_flag(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED", raising=False)
    monkeypatch.setattr("typesafe_rag_guard.typesafe_available", lambda _env=None: True)

    assert alignment.alignment_enabled({"TYPESAFE_API_KEY": "x"}) is False
    assert alignment.alignment_enabled(
        {"TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED": "1", "TYPESAFE_API_KEY": "x"}
    ) is True
