"""Lost reason and competitor field normalizers shared by batch/DB jobs."""

from __future__ import annotations


CANONICAL_LOST_REASONS = {
    "",
    "他社競合（レート）",
    "他社競合（その他）",
    "調達方法変更",
    "設備見合わせ",
    "物件不適",
    "業績不振",
    "その他（不成約）",
    "理由未入力",
}

FUNDING_NOT_COMPETITOR = {"現金", "銀行借入対応", "融資対応", "自己資金"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().replace(" ", "").replace("　", "")


def normalize_lost_reason(raw_value: object, final_status: str = "") -> str:
    raw = clean_text(raw_value)
    status = clean_text(final_status)

    if raw in {"", "0", "０", "none", "None", "null", "NULL", "未入力", "不明"}:
        return "理由未入力" if status == "失注" else ""

    if "業績不振" in raw:
        return "業績不振"
    if "中古物件" in raw or "物件不適" in raw or "不可" in raw:
        return "物件不適"
    if "他社" in raw or "他行" in raw or "他リース" in raw:
        if "レート" in raw or "金利" in raw or "%" in raw or "利率" in raw:
            return "他社競合（レート）"
        return "他社競合（その他）"
    if "金利" in raw or "レート" in raw:
        return "他社競合（レート）"
    if (
        "見合" in raw
        or "見させ" in raw
        or "見出せ" in raw
        or "延期" in raw
        or "延用" in raw
        or "投資" in raw
    ):
        return "設備見合わせ"
    if (
        "方法" in raw
        or "自己資金" in raw
        or "自己貴金" in raw
        or "自己責金" in raw
        or "自己金" in raw
        or "白己" in raw
        or "己資" in raw
        or "現金" in raw
        or "融資対応" in raw
    ):
        return "調達方法変更"
    if raw in CANONICAL_LOST_REASONS:
        return raw
    return "その他（不成約）"


def normalize_competitor_fields(
    competitor: object,
    competitor_name: object = "",
    lost_reason: object = "",
) -> tuple[str, str]:
    raw_competitor = clean_text(competitor)
    raw_name = clean_text(competitor_name)
    reason = clean_text(lost_reason)

    if raw_competitor == "競合あり":
        normalized_competitor = "競合あり"
    elif raw_competitor == "競合なし":
        normalized_competitor = "競合なし"
    elif reason.startswith("他社競合"):
        normalized_competitor = "競合あり"
    elif raw_name and raw_name not in {"0", "０"} and raw_name not in FUNDING_NOT_COMPETITOR:
        normalized_competitor = "競合あり"
    else:
        normalized_competitor = "競合なし"

    normalized_name = "" if raw_name in {"", "0", "０"} else raw_name
    if normalized_competitor == "競合なし" and normalized_name in FUNDING_NOT_COMPETITOR:
        normalized_name = ""
    return normalized_competitor, normalized_name
