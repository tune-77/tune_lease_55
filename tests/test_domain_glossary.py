"""P7-001 リース業界ドメイン辞書ローダーの単体テスト（AC-701〜705）。"""
import json

from api.knowledge.domain_glossary import get_glossary, known_terms, synonyms_for


def test_ac701_seed_glossary_loads():
    glossary = get_glossary()
    assert len(glossary["synonym_groups"]) >= 8
    assert len(glossary["industry_terms"]) >= 5
    for group in glossary["synonym_groups"]:
        assert group["canonical"]
        assert group["synonyms"]
        assert 0.0 <= group["weight"] <= 1.0
        assert group["source"]


def test_ac702_synonym_lookup_is_bidirectional():
    from_canonical = [term for term, _w in synonyms_for("赤字企業")]
    from_synonym = [term for term, _w in synonyms_for("営業赤字")]
    assert "営業赤字" in from_canonical
    assert "赤字企業" in from_synonym


def test_ac703_missing_file_falls_back_to_empty(tmp_path):
    missing = str(tmp_path / "no_such_glossary.json")
    glossary = get_glossary(path=missing)
    assert glossary["synonym_groups"] == []
    assert glossary["industry_terms"] == []
    assert synonyms_for("赤字企業", path=missing) == []
    assert known_terms(path=missing) == frozenset()


def test_ac703_broken_json_falls_back_to_empty(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{ this is not json", encoding="utf-8")
    assert get_glossary(path=str(broken))["synonym_groups"] == []


def test_ac704_lookup_normalizes_case_and_width():
    expected = synonyms_for("LTV")
    assert expected, "シード辞書に LTV グループがある前提"
    assert synonyms_for("ltv") == expected
    assert synonyms_for("ＬＴＶ") == expected


def test_ac705_reload_on_mtime_change(tmp_path):
    path = tmp_path / "glossary.json"

    def _write(canonical: str, mtime: float):
        path.write_text(json.dumps({
            "version": 1,
            "updated": "2026-07-09",
            "synonym_groups": [
                {"canonical": canonical, "synonyms": ["別語"], "weight": 1.0, "source": "test"}
            ],
            "industry_terms": [],
        }, ensure_ascii=False), encoding="utf-8")
        import os
        os.utime(path, (mtime, mtime))

    _write("語A", 1_000_000)
    assert get_glossary(path=str(path))["synonym_groups"][0]["canonical"] == "語A"
    _write("語B", 2_000_000)
    assert get_glossary(path=str(path))["synonym_groups"][0]["canonical"] == "語B"


def test_invalid_weight_is_clamped(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({
        "version": 1,
        "updated": "2026-07-09",
        "synonym_groups": [
            {"canonical": "語A", "synonyms": ["別語"], "weight": 1.7, "source": "test"},
            {"canonical": "語B", "synonyms": [], "weight": 1.0, "source": "test"},
        ],
        "industry_terms": [],
    }, ensure_ascii=False), encoding="utf-8")
    glossary = get_glossary(path=str(path))
    assert len(glossary["synonym_groups"]) == 1  # synonyms 空のグループはスキップ
    assert glossary["synonym_groups"][0]["weight"] == 1.0


def test_known_terms_includes_industry_aliases():
    terms = known_terms()
    assert "Q-Risk" in terms
    assert "Qリスク" in terms
    assert "建機" in terms
