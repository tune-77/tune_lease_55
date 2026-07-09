"""P7-002 クエリ拡張の単体テスト（AC-711・AC-712 ほか）。シード辞書を使用する。"""
from api.knowledge.query_expansion import expand_query


def test_ac711_basic_expansion():
    expanded = expand_query("飲食業の赤字企業")
    assert expanded[0] == {"query": "飲食業の赤字企業", "weight": 1.0, "replaced": ""}
    variants = [e["query"] for e in expanded[1:]]
    assert "飲食店の赤字企業" in variants
    assert "飲食業の営業赤字" in variants
    assert len(expanded) <= 5  # 元クエリ + max_variants(4)
    for entry in expanded[1:]:
        assert 0.0 < entry["weight"] < 1.0
        assert entry["replaced"] in ("飲食業", "赤字企業")


def test_ac711_construction_machine():
    variants = [e["query"] for e in expand_query("建設機械リース")[1:]]
    assert "建機リース" in variants or "重機リース" in variants


def test_ac712_no_match_is_noop():
    expanded = expand_query("今日の天気")
    assert expanded == [{"query": "今日の天気", "weight": 1.0, "replaced": ""}]


def test_max_variants_zero_returns_original_only():
    expanded = expand_query("飲食業の赤字企業", max_variants=0)
    assert len(expanded) == 1
    assert expanded[0]["weight"] == 1.0


def test_empty_query_is_safe():
    expanded = expand_query("")
    assert len(expanded) == 1


def test_decay_scales_weight():
    strong = expand_query("残価の考え方", decay=0.5)
    weak = expand_query("残価の考え方", decay=0.3)
    assert len(strong) > 1 and len(weak) > 1
    assert strong[1]["weight"] > weak[1]["weight"]


def test_overlapping_terms_do_not_produce_broken_queries():
    # 「赤字」は「赤字企業」の部分文字列。壊れた置換（例: 赤字企業企業）が出ないこと
    variants = [e["query"] for e in expand_query("飲食業の赤字企業", max_variants=8)]
    assert not any("企業企業" in v for v in variants)


def test_missing_glossary_is_noop(tmp_path):
    expanded = expand_query("飲食業の赤字企業", path=str(tmp_path / "none.json"))
    assert len(expanded) == 1
