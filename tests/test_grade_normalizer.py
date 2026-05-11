import pytest

from grade_normalizer import is_excluded_grade, normalize_grade


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", "1-3"),
        ("2", "1-3"),
        ("3", "1-3"),
        ("①1-3 (優良)", "1-3"),
        ("1〜3", "1-3"),
        ("4", "4-6"),
        ("5", "4-6"),
        ("6", "4-6"),
        ("②4-6 (標準)", "4-6"),
        ("4－6", "4-6"),
        ("8_1要注意先", "要注意先"),
        ("8_2要注意先", "要注意先"),
        ("③要注意以下", "要注意先"),
        ("7-9", "要注意先"),
        ("無格付", "無格付"),
        ("④無格付", "無格付"),
        ("", "無格付"),
        (None, "無格付"),
        ("不明", "無格付"),
    ],
)
def test_normalize_grade(raw, expected):
    assert normalize_grade(raw) == expected


@pytest.mark.parametrize("raw", ["8-3", "8_3要注意先", "9", "9(要注意)", "10", "10(破綻懸念)"])
def test_is_excluded_grade(raw):
    assert is_excluded_grade(raw)


@pytest.mark.parametrize("raw", ["8_1要注意先", "8_2要注意先", "要注意先", "1-3", "4-6", "無格付"])
def test_is_not_excluded_grade(raw):
    assert not is_excluded_grade(raw)
