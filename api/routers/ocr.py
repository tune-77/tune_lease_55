import base64
import json
import os
import re
from typing import Optional

import requests
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.cloudrun_writeback import record_cloudrun_input_event
from api.llm_json_guard import extract_candidate_text, parse_or_recover_json, with_retry_tokens

router = APIRouter()

_OCR_SUPPORTED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
_PII_REDACTION_TOKEN = "[REDACTED]"
_PII_PERSON_TOKEN = "[REDACTED_PERSON]"
_PII_REGEXES = [
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{2,4}-\d{2,4}-\d{3,4}\b"),
    re.compile(r"\b\d{3}-\d{4}\b"),
    re.compile(r"\b\d{12}\b"),
]


def _get_api_key() -> str:
    try:
        from secret_manager import get_gemini_api_key  # type: ignore

        val = get_gemini_api_key()
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    val = os.environ.get("GEMINI_API_KEY", "")
    return val.strip() if isinstance(val, str) else ""


def _ocr_url() -> str:
    model = (os.environ.get("GEMINI_OCR_MODEL") or "gemini-2.0-flash").strip()
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


_FINANCIAL_OCR_FIELDS = [
    "nenshu",
    "gross_profit",
    "op_profit",
    "ord_profit",
    "net_income",
    "net_assets",
    "total_assets",
    "depreciation",
    "dep_expense",
    "rent",
    "rent_expense",
    "machines",
    "other_assets",
]


_OCR_PROMPT = """あなたは財務諸表OCR専門AIです。添付の画像またはPDFから財務指標を読み取り、短い構造JSONのみを返してください。

抽出フィールド（すべて百万円単位、見つからない場合はnull）:
{
  "nenshu": <売上高>,
  "gross_profit": <売上総利益（粗利）>,
  "op_profit": <営業利益>,
  "ord_profit": <経常利益>,
  "net_income": <当期純利益（最終損益）>,
  "net_assets": <純資産（自己資本合計）>,
  "total_assets": <総資産（資産合計）>,
  "depreciation": <減価償却費（B/S計上額）>,
  "dep_expense": <減価償却費（P/L費用）>,
  "rent": <賃借料（B/S使用権資産）>,
  "rent_expense": <賃借料（P/L費用）>,
  "machines": <機械装置・運搬具・車両の帳簿価額>,
  "other_assets": <その他固定資産合計>,
  "detected_fields": ["値を読み取れたフィールド名だけを英字キーで列挙"],
  "missing_fields": ["読み取れなかったフィールド名だけを英字キーで列挙"],
  "confidence": <全体の読み取り信頼度 0.0〜1.0>
}

重要ルール:
1. 赤字・損失は必ず負値で返す（▲500 → -500、△2,000 → -2000）。負値をゼロや正値に変換しない
2. 元の書類が「円」単位なら100万で除算、「千円」単位なら1000で除算して百万円に統一
3. 複数期のデータがある場合は最新期（当期）の値を使用
4. 値が不明・記載なしの場合はnullを返す（実際にゼロの場合のみ0を使う）
5. detected_fields / missing_fields は上記の英字キーのみを使う
6. 個人名、住所、電話番号、メールアドレス、マイナンバーなど個人特定情報は抽出せず、判断に必要な財務数値だけを返す
7. JSON以外の文字列は絶対に返さない"""


_OCR_TAX_CERT_PROMPT = """あなたは納税証明書OCR専門AIです。添付の画像またはPDFから納税証明書（滞納なし確認・税額情報）を読み取り、JSONのみを返してください。

抽出フィールド:
{
  "tax_default": <滞納有無フラグ（滞納がない場合はfalse、滞納がある場合はtrue）>,
  "tax_amount": <税額（整数、見つからない場合はnull）>
}

重要ルール:
1. 滞納の有無を正確に判断し、ブール値で返す。
2. 税額は、元の書類が「円」単位ならそのまま、「千円」単位なら1000倍、「万円」単位なら10000倍して「円」単位の整数に統一する。
3. 税額が見つからない場合はnullを返す。
4. JSON以外の文字列は絶対に返さない。"""


_OCR_TOUKIBO_PROMPT = """あなたは登記簿謄本OCR専門AIです。添付の画像またはPDFから登記簿謄本情報を読み取り、JSONのみを返してください。

抽出フィールド:
{
  "established_date": <設立年月日（YYYY-MM-DD形式の文字列）>,
  "capital": <資本金（円単位の整数、見つからない場合はnull）>,
  "representative": <代表者名（文字列）>,
  "directors": <役員名リスト（文字列の配列、見つからない場合は空の配列）>
}

重要ルール:
1. 設立年月日は「YYYY-MM-DD」形式の文字列で返す。
2. 資本金は、元の書類が「円」単位ならそのまま、「千円」単位なら1000倍、「万円」単位なら10000倍して「円」単位の整数に統一する。
3. 資本金が見つからない場合はnullを返す。
4. 代表者名・役員名など個人名は返さず、代表者の記載がある場合も代表者名は"[REDACTED_PERSON]"にする。
5. 役員名リストは個人特定情報を含むため空配列で返す。
6. JSON以外の文字列は絶対に返さない。"""


_OCR_ESTIMATE_PROMPT = """あなたは見積書・注文書OCR専門AIです。添付の画像またはPDFから見積書・注文書情報を読み取り、JSONのみを返してください。

抽出フィールド:
{
  "item_name": <物件名または品名（文字列）>,
  "amount": <合計金額（円単位の整数、見つからない場合はnull）>,
  "maker": <メーカー名（文字列）>,
  "model_number": <型番（文字列）>
}

重要ルール:
1. 物件名または品名は、書類全体から最も適切と思われる主要な名称を文字列で返す。
2. 合計金額は、元の書類が「円」単位ならそのまま、「千円」単位なら1000倍、「万円」単位なら10000倍して「円」単位の整数に統一する。消費税込みの金額を優先する。
3. 金額が見つからない場合はnullを返す。
4. メーカー名と型番は、書類に記載されている場合は文字列で返す。見つからない場合は空文字列を返す。
5. 個人名、住所、電話番号、メールアドレスなど個人特定情報は返さない。
6. JSON以外の文字列は絶対に返さない。"""


_OCR_BROCHURE_PROMPT = """あなたは会社案内・パンフレットOCR専門AIです。添付の画像またはPDFから会社案内・パンフレット情報を読み取り、JSONのみを返してください。

抽出フィールド:
{
  "industry": <業種（文字列）>,
  "business_description": <事業内容（文字列）>,
  "main_clients": <主要取引先リスト（文字列の配列、見つからない場合は空の配列）>,
  "employee_count": <従業員数（整数、見つからない場合はnull）>
}

重要ルール:
1. 業種は、会社が属する主要な業界を文字列で返す。
2. 事業内容は、会社の主要な事業活動を簡潔な文字列で返す。
3. 主要取引先は、会社が特に強調している取引先を文字列の配列として返す。見つからない場合は空の配列を返す。
4. 従業員数は、書類に記載されている場合は整数で返す。おおよその人数（例: 100名程度）の場合も整数に変換する。見つからない場合はnullを返す。
5. JSON以外の文字列は絶対に返さない。"""


def _ocr_defaults(doc_type: Optional[str]) -> dict:
    if doc_type == "tax_cert":
        return {"tax_default": None, "tax_amount": None}
    if doc_type == "toukibo":
        return {"established_date": "", "capital": None, "representative": "", "directors": []}
    if doc_type == "estimate":
        return {"item_name": "", "amount": None, "maker": "", "model_number": ""}
    if doc_type == "brochure":
        return {"industry": "", "business_description": "", "main_clients": [], "employee_count": None}
    financial_defaults = {
        "nenshu": None,
        "gross_profit": None,
        "op_profit": None,
        "ord_profit": None,
        "net_income": None,
        "net_assets": None,
        "total_assets": None,
        "depreciation": None,
        "dep_expense": None,
        "rent": None,
        "rent_expense": None,
        "machines": None,
        "other_assets": None,
    }
    return {
        **financial_defaults,
        "detected_fields": [],
        "missing_fields": list(_FINANCIAL_OCR_FIELDS),
        "confidence": 0.0,
    }


def _ocr_field_groups(doc_type: Optional[str]) -> tuple[set[str], set[str], set[str], set[str]]:
    if doc_type == "tax_cert":
        return set(), set(), {"tax_amount"}, {"tax_default"}
    if doc_type == "toukibo":
        return {"established_date", "representative"}, {"directors"}, {"capital"}, set()
    if doc_type == "estimate":
        return {"item_name", "maker", "model_number"}, set(), {"amount"}, set()
    if doc_type == "brochure":
        return {"industry", "business_description"}, {"main_clients"}, {"employee_count"}, set()
    return set(), {"detected_fields", "missing_fields"}, set(_FINANCIAL_OCR_FIELDS) | {"confidence"}, set()


def _redact_pii_text(value: str) -> tuple[str, bool]:
    redacted = value
    changed = False
    for pattern in _PII_REGEXES:
        next_value = pattern.sub(_PII_REDACTION_TOKEN, redacted)
        if next_value != redacted:
            changed = True
            redacted = next_value
    return redacted, changed


def _redact_ocr_pii(result: dict, doc_type: Optional[str]) -> dict:
    redacted_fields: set[str] = set()

    if doc_type == "toukibo":
        if result.get("representative"):
            result["representative"] = _PII_PERSON_TOKEN
            redacted_fields.add("representative")
        if result.get("directors"):
            result["directors"] = []
            redacted_fields.add("directors")

    for key, value in list(result.items()):
        if isinstance(value, str):
            redacted_value, changed = _redact_pii_text(value)
            if changed:
                result[key] = redacted_value
                redacted_fields.add(key)
        elif isinstance(value, list):
            next_values = []
            changed = False
            for item in value:
                if isinstance(item, str):
                    redacted_item, item_changed = _redact_pii_text(item)
                    next_values.append(redacted_item)
                    changed = changed or item_changed
                else:
                    next_values.append(item)
            if changed:
                result[key] = next_values
                redacted_fields.add(key)

    result["pii_redacted_fields"] = sorted(redacted_fields)
    return result


def _normalize_ocr_result(result: dict, doc_type: Optional[str]) -> dict:
    if doc_type == "tax_cert":
        if result.get("tax_default") is not None and not isinstance(result.get("tax_default"), bool):
            raise ValueError("tax_default must be a boolean or null.")
        if result.get("tax_amount") is not None and type(result.get("tax_amount")) is not int:
            raise ValueError("tax_amount must be an integer or null.")
    elif doc_type == "toukibo":
        if not isinstance(result.get("established_date"), str):
            raise ValueError("established_date must be a string.")
        if result.get("capital") is not None and type(result.get("capital")) is not int:
            raise ValueError("capital must be an integer or null.")
        if not isinstance(result.get("representative"), str):
            raise ValueError("representative must be a string.")
        if not isinstance(result.get("directors"), list):
            raise ValueError("directors must be a list.")
        result["directors"] = [director for director in result.get("directors", []) if isinstance(director, str)]
    elif doc_type == "estimate":
        if not isinstance(result.get("item_name"), str):
            raise ValueError("item_name must be a string.")
        if result.get("amount") is not None and type(result.get("amount")) is not int:
            raise ValueError("amount must be an integer or null.")
        if not isinstance(result.get("maker"), str):
            raise ValueError("maker must be a string.")
        if not isinstance(result.get("model_number"), str):
            raise ValueError("model_number must be a string.")
    elif doc_type == "brochure":
        if not isinstance(result.get("industry"), str):
            raise ValueError("industry must be a string.")
        if not isinstance(result.get("business_description"), str):
            raise ValueError("business_description must be a string.")
        if not isinstance(result.get("main_clients"), list):
            raise ValueError("main_clients must be a list.")
        result["main_clients"] = [client for client in result.get("main_clients", []) if isinstance(client, str)]
        if result.get("employee_count") is not None and type(result.get("employee_count")) is not int:
            raise ValueError("employee_count must be an integer or null.")
    else:
        detected_fields = [
            key for key in _FINANCIAL_OCR_FIELDS
            if result.get(key) is not None and type(result.get(key)) in (int, float)
        ]
        missing_fields = [key for key in _FINANCIAL_OCR_FIELDS if key not in detected_fields]
        for key in _FINANCIAL_OCR_FIELDS:
            if result.get(key) is not None and type(result.get(key)) not in (int, float):
                raise ValueError(f"{key} must be a number or null.")
        confidence = result.get("confidence")
        if type(confidence) not in (int, float):
            confidence = round(len(detected_fields) / len(_FINANCIAL_OCR_FIELDS), 2)
        confidence = max(0.0, min(1.0, float(confidence)))
        if not detected_fields:
            confidence = 0.0
        result["detected_fields"] = detected_fields
        result["missing_fields"] = missing_fields
        result["confidence"] = round(confidence, 2)
    return _redact_ocr_pii(result, doc_type)


@router.post("/ocr")
async def ocr_financial(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: Optional[str] = None,
) -> JSONResponse:
    if file.content_type not in _OCR_SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="サポートされていないファイル形式です。対応形式: JPEG, PNG, GIF, WebP, PDF",
        )

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=400, detail="ファイルサイズが20MBを超えています")

    api_key = _get_api_key()
    if not api_key:
        return JSONResponse(
            status_code=503,
            content={"error": "GEMINI_API_KEY が設定されていません"},
        )

    file_b64 = base64.b64encode(content).decode("utf-8")

    current_prompt = _OCR_PROMPT  # デフォルトは財務諸表OCR

    if doc_type == "tax_cert":
        current_prompt = _OCR_TAX_CERT_PROMPT
    elif doc_type == "toukibo":
        current_prompt = _OCR_TOUKIBO_PROMPT
    elif doc_type == "estimate":
        current_prompt = _OCR_ESTIMATE_PROMPT
    elif doc_type == "brochure":
        current_prompt = _OCR_BROCHURE_PROMPT

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": file.content_type, "data": file_b64}},
                    {"text": current_prompt},
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1, "maxOutputTokens": 2048},
    }

    try:
        defaults = _ocr_defaults(doc_type)
        string_fields, array_fields, number_fields, bool_fields = _ocr_field_groups(doc_type)
        result = defaults
        finish_reason = ""
        for current_payload in (payload, with_retry_tokens(payload, 4096)):
            resp = requests.post(
                _ocr_url(),
                json=current_payload,
                headers={"x-goog-api-key": api_key},
                timeout=60,
            )
            resp.raise_for_status()
            text, finish_reason = extract_candidate_text(resp.json())
            result, recovered = parse_or_recover_json(
                text,
                defaults=defaults,
                string_fields=string_fields,
                array_fields=array_fields,
                number_fields=number_fields,
                bool_fields=bool_fields,
            )
            if not recovered and finish_reason != "MAX_TOKENS":
                break
            if current_payload["generationConfig"]["maxOutputTokens"] >= 4096:
                break
        if finish_reason == "MAX_TOKENS":
            result["_finish_reason"] = finish_reason
        normalized = _normalize_ocr_result(result, doc_type)
        background_tasks.add_task(
            record_cloudrun_input_event,
            event_type="ocr_extracted",
            surface="ocr",
            payload={
                "schema_version": 1,
                "doc_type": doc_type or "financial",
                "content_type": file.content_type or "",
                "result": normalized,
            },
        )
        return JSONResponse(content=normalized)
    except requests.HTTPError as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini API エラー: {exc}"})
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini レスポンス解析エラーまたは検証エラー: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"OCR処理に失敗しました: {exc}"})
