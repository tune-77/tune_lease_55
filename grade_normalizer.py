"""Credit grade normalization helpers."""


GRADE_1_3 = "1-3"
GRADE_4_6 = "4-6"
GRADE_WATCH = "要注意先"
GRADE_NONE = "無格付"


def is_excluded_grade(value) -> bool:
    """Return True for grades excluded from analysis/input data."""
    s = str(value or "").strip()
    compact = (
        s.replace("－", "-")
        .replace("ー", "-")
        .replace("―", "-")
        .replace("〜", "-")
        .replace("～", "-")
        .replace("_", "-")
    )
    return compact.startswith("8-3") or compact.startswith("9") or compact.startswith("10")


def normalize_grade(value) -> str:
    """Normalize known grade notations into 4 categories."""
    s = str(value or "").strip()
    if not s or s in {"未設定", "None", "nan"}:
        return GRADE_NONE
    if "無格付" in s:
        return GRADE_NONE
    if (
        "要注意" in s
        or s.startswith("8")
        or s.startswith("9")
        or "7-9" in s
        or "7〜9" in s
        or "7－9" in s
    ):
        return GRADE_WATCH
    if "1-3" in s or "1〜3" in s or "1－3" in s or s in {"1", "2", "3"}:
        return GRADE_1_3
    if "4-6" in s or "4〜6" in s or "4－6" in s or s in {"4", "5", "6"}:
        return GRADE_4_6
    return GRADE_NONE
