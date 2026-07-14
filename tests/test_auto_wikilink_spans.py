from scripts.auto_wikilink import linkify_body


def test_auto_wikilink_does_not_link_inside_existing_link_with_brackets():
    body = "- 参照ノート: [[05-クリップ_記事/リースニュース/2026-07-11_芙蓉総合リース[8424]_静岡市.md]]"
    title_index = {"2026-07-11": "2026-07-11"}

    new_body, changes = linkify_body(body, title_index, own_stem="source")

    assert changes == 0
    assert new_body == body
