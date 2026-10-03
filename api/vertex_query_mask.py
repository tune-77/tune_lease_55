"""Vertex AI Search / Answer API へ送るクエリから審査データの個人情報を伏せる。

チャット本文はそのまま外部の検索APIへ出ていたため、送る直前に必ず通す。
会社名・個人名・電話・メール・郵便番号・住所・金額・「氏名：」等のラベル付き値を置き換える。
業種・物件・スコア・年数など、検索に必要な審査概念は残す（ベストエフォートで完全除去は保証しない）。
パターンは api/crystallizer/bias_extractor.py と jev_safe_gateway.py の既存規則に揃えている。
"""

from __future__ import annotations

import re

_NAME_TOKEN = r"[一-龠ァ-ヶーA-Za-z0-9]{1,20}"
_COMPANY_RE = re.compile(
    rf"(?:株式会社|有限会社|合同会社|合資会社|\(株\)|（株）|\(有\)|（有）)\s*(?:{_NAME_TOKEN})?"
    rf"|(?:{_NAME_TOKEN})(?:株式会社|有限会社|合同会社|合資会社|\(株\)|（株）)"
)
_PERSON_RE = re.compile(rf"(?:{_NAME_TOKEN})(?:様|さん|氏|社長|専務|常務|部長|課長|代表)")
_LABELED_RE = re.compile(
    r"(氏名|名前|住所|所在地|電話番号|電話|TEL|携帯|メール|顧客名|企業名|法人名|取引先名|会社名|社名|商号|申込者|代表者名|代表者|担当者)\s*[:：]\s*[^\s、。,，\n]+",
    re.I,
)
# 自由記述欄（紫苑レビュー依頼の「営業メモ:」等）は人名・社名・経緯が混ざるので行末まで伏せる
_FREE_TEXT_LABELED_RE = re.compile(r"(営業メモ|現場メモ|担当者メモ|備考|特記事項)\s*[:：][^\n]*")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\d)(?:0\d{1,4}[-ー‐(（]?\d{1,4}[-ー‐)）]?\d{3,4})(?!\d)")
_POSTAL_RE = re.compile(r"〒\s?\d{3}[-ー‐]\d{4}|(?<![\d\-ー‐])\d{3}[-ー‐]\d{4}(?![\d\-ー‐])")
_ADDRESS_RE = re.compile(
    r"(?:東京都|北海道|(?:京都|大阪)府|[一-龠]{2,3}県)[一-龠ァ-ヶぁ-んー]{1,12}?[市区町村郡]"
    r"(?:[一-龠ァ-ヶぁ-んー0-9０-９]{0,20}?(?:丁目|番地|番|号|-|ー))*[0-9０-９\-ー]*"
)
# 金額（審査データ）。「3年」「65点」「70%」等の審査概念は対象外
_MONEY_RE = re.compile(
    r"[¥￥]\s?[0-9０-９,，.]+(?:万|百万|千万|億)?円?"
    r"|[0-9０-９][0-9０-９,，.]*\s?(?:兆|億|千万|百万|万|千)?\s?円"
    r"|[0-9０-９][0-9０-９,，.]*\s?(?:兆|億|千万|百万|万)(?=\s|$|[、。,，の])"
)


def mask_for_vertex(text: str) -> str:
    masked = str(text or "")
    masked = _FREE_TEXT_LABELED_RE.sub(lambda m: f"{m.group(1)}:〈伏字〉", masked)
    masked = _LABELED_RE.sub(lambda m: f"{m.group(1)}:〈伏字〉", masked)
    masked = _EMAIL_RE.sub("〈メール〉", masked)
    masked = _PHONE_RE.sub("〈電話〉", masked)  # 郵便番号より先（電話の後半7桁を郵便番号と誤認しない）
    masked = _POSTAL_RE.sub("〈郵便番号〉", masked)
    masked = _ADDRESS_RE.sub("〈住所〉", masked)
    masked = _MONEY_RE.sub("〈金額〉", masked)
    masked = _COMPANY_RE.sub("〈企業〉", masked)
    masked = _PERSON_RE.sub("〈人物〉", masked)
    return masked
