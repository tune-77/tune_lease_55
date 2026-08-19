from pathlib import Path

from scripts.obsidian_theme_radar import build_theme_radar


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_finds_buzzing_theme_and_proposes_angle(tmp_path):
    vault = tmp_path
    for i in range(3):
        _write(vault / f"note{i}.md", "---\ntags: [機械学習]\n---\n機械学習の勉強メモ。")
    _write(vault / "note_other.md", "---\ntags: [料理]\n---\n料理のメモ。")

    def fake_search(query, **kwargs):
        if "機械学習" in query:
            return {
                "used": True,
                "status": "ok",
                "text": "機械学習が話題です",
                "sources": [{"title": "A", "uri": "http://a"}, {"title": "B", "uri": "http://b"}],
            }
        return {"used": True, "status": "ok", "text": "", "sources": []}

    def fake_generate(prompt, **kwargs):
        return {"used": True, "status": "ok", "text": "機械学習×学習ログの切り口", "error": ""}

    report = build_theme_radar(vault, search_fn=fake_search, generate_fn=fake_generate)

    assert report["recurring_themes"][0]["theme"] == "機械学習"
    assert report["selected_theme"] == "機械学習"
    assert report["proposed_angle"]["angle"] == "機械学習×学習ログの切り口"


def test_no_buzz_still_proposes_angle_for_top_theme(tmp_path):
    vault = tmp_path
    _write(vault / "note.md", "---\ntags: [料理]\n---\n料理のメモ。")

    def fake_search(query, **kwargs):
        return {"used": True, "status": "ok", "text": "", "sources": []}

    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(prompt)
        return {"used": True, "status": "ok", "text": "料理×システム改善の切り口", "error": ""}

    report = build_theme_radar(vault, search_fn=fake_search, generate_fn=fake_generate)

    assert report["selected_theme"] == "料理"
    assert report["proposed_angle"]["status"] == "ok"
    assert report["proposed_angle"]["angle"] == "料理×システム改善の切り口"
    assert len(calls) == 1
    assert "リース審査AI" in calls[0]


def test_empty_vault_yields_no_selected_theme(tmp_path):
    vault = tmp_path
    vault.mkdir(exist_ok=True)

    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(prompt)
        return {"used": True, "status": "ok", "text": "should not be called", "error": ""}

    report = build_theme_radar(
        vault,
        search_fn=lambda *a, **k: {"used": True, "status": "ok", "text": "", "sources": []},
        generate_fn=fake_generate,
    )

    assert report["selected_theme"] == ""
    assert report["proposed_angle"]["status"] == "no_theme"
    assert calls == []


def test_frontmatter_boilerplate_does_not_pollute_themes(tmp_path):
    vault = tmp_path
    _write(vault / "note.md", "---\ntags: [固有テーマ]\n---\n固有テーマについてのメモ。")

    def fake_search(query, **kwargs):
        return {"used": True, "status": "ok", "text": "", "sources": []}

    report = build_theme_radar(vault, search_fn=fake_search, generate_fn=lambda *a, **k: {"used": False})

    themes = {item["theme"] for item in report["recurring_themes"]}
    assert "tags" not in themes
