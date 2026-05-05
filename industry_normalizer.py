"""Industry label normalization helpers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


CANONICAL_INDUSTRY_MAJOR = [
    "A 農業・林業",
    "B 漁業",
    "C 鉱業・採石業・砂利採取業",
    "D 建設業",
    "E 製造業",
    "F 電気・ガス・熱供給・水道業",
    "G 情報通信業",
    "H 運輸業・郵便業",
    "I 卸売業・小売業",
    "J 金融業・保険業",
    "K 不動産業・物品賃貸業",
    "L 学術研究・専門・技術サービス業",
    "M 宿泊業・飲食サービス業",
    "N 生活関連サービス業・娯楽業",
    "O 教育・学習支援業",
    "P 医療・福祉",
    "Q 複合サービス事業",
    "R サービス業(他に分類されないもの)",
]


_CODE_TO_MAJOR = {label.split(" ", 1)[0]: label for label in CANONICAL_INDUSTRY_MAJOR}


@lru_cache(maxsize=1)
def _load_major_to_default_sub() -> dict[str, str]:
    """Load a representative industry-subcategory per major category.

    The batch workflow often receives only the major industry classification.
    For downstream screens that expect a subcategory, we pick the first
    benchmark key available for that major.
    """
    candidates = [
        Path(__file__).resolve().parent / "static_data" / "industry_benchmarks.json",
        Path(__file__).resolve().parent / "data" / "industry_benchmarks.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mapping: dict[str, str] = {}
        for major_label, payload in data.items():
            if not isinstance(payload, dict):
                continue
            sub_map = payload.get("sub") or {}
            if not sub_map:
                continue
            major_code = str(major_label).strip().split(" ", 1)[0][:1]
            if major_code and major_code not in mapping:
                mapping[major_code] = next(iter(sub_map.keys()))
        if mapping:
            return mapping
    return {
        "D": "06 総合工事業",
        "E": "09 食料品製造業",
        "H": "44 道路貨物運送業",
        "I": "50-55 各種卸売業",
        "J": "64 金融業",
        "K": "68 不動産代理・仲介",
        "M": "75 宿泊業",
        "P": "83 医療業(病院・診療所)",
        "R": "91 職業紹介・労働者派遣業",
    }


def _compact(value: str) -> str:
    text = str(value or "").strip()
    text = text.translate(str.maketrans({
        "（": "(",
        "）": ")",
        "〔": "(",
        "〕": ")",
        "　": " ",
    }))
    text = re.sub(r"\s+", "", text)
    text = text.replace("･", "・").replace(" ", "")
    return text.upper()


def normalize_industry_major(value: str) -> str:
    """Return a canonical JSIC major industry label for common OCR variants."""
    raw = str(value or "").strip()
    if not raw:
        return ""

    compact = _compact(raw)
    if len(compact) >= 2 and compact[0] in _CODE_TO_MAJOR:
        code = compact[0]
        if code == "P" and "医療" in compact:
            return _CODE_TO_MAJOR[code]
        if code == "R" and "サービス" in compact:
            return _CODE_TO_MAJOR[code]
        if code not in {"P", "R"}:
            return _CODE_TO_MAJOR[code]

    # OCR often drops the JSIC code or corrupts only the leading code.
    keyword_map = [
        (("建設",), "D"),
        (("製造",), "E"),
        (("電気", "ガス"), "F"),
        (("情報通信",), "G"),
        (("運輸",), "H"),
        (("郵便",), "H"),
        (("卸売",), "I"),
        (("小売",), "I"),
        (("金融",), "J"),
        (("保険",), "J"),
        (("不動産",), "K"),
        (("物品賃貸",), "K"),
        (("宿泊",), "M"),
        (("飲食",), "M"),
        (("医療",), "P"),
        (("福祉",), "P"),
        (("複合サービス",), "Q"),
        (("サービス業",), "R"),
    ]
    for keywords, code in keyword_map:
        if any(keyword in compact for keyword in keywords):
            return _CODE_TO_MAJOR[code]

    return raw


def _extract_major_code(value: str) -> str:
    major = normalize_industry_major(value)
    if not major:
        return ""
    return major.split(" ", 1)[0][:1]


def normalize_industry_sub(value: str, major_value: str = "") -> str:
    """Normalize an industry-subcategory field.

    Treat blank-like values such as 0 as missing. If the subcategory is absent,
    fall back to a representative benchmark subcategory derived from the major
    industry.
    """
    raw = str(value or "").strip()
    if raw and raw not in {"0", "0.0", "nan", "None"}:
        return raw

    major_code = _extract_major_code(major_value)
    if major_code:
        return _load_major_to_default_sub().get(major_code, "")
    return ""
