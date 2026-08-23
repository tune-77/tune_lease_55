from scripts.evaluate_rag_retrieval_variants import (
    _category_boosted,
    _filter_excluded,
    _unique_by_path,
)


def test_variant_helpers_remove_noise_and_dedupe_paths():
    hits = [
        {"file_path": "Projects/tune_lease_55/AI Chat/2026-07-01.md", "rank_score": 1.0},
        {"file_path": "03-知識_業界/リース基礎知識/ファイナンスリース.md", "rank_score": 0.8},
        {"file_path": "03-知識_業界/リース基礎知識/ファイナンスリース.md", "rank_score": 0.7},
    ]

    filtered = _filter_excluded(hits, ("Projects/tune_lease_55/AI Chat/",))
    unique = _unique_by_path(filtered, top_k=5)

    assert [hit["file_path"] for hit in unique] == [
        "03-知識_業界/リース基礎知識/ファイナンスリース.md"
    ]


def test_category_boost_prefers_matching_path_hint():
    hits = [
        {"file_path": "Projects/tune_lease_55/Research/general.md", "rank_score": 0.9},
        {"file_path": "Projects/tune_lease_55/Asset Knowledge/医療機器/医療機器 残価・保守期限・薬機法.md", "rank_score": 0.3},
    ]

    boosted = _category_boosted(hits, "asset_medical")

    assert boosted[0]["file_path"].startswith("Projects/tune_lease_55/Asset Knowledge/医療機器/")
