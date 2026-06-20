import base64
import json
import os
from typing import Optional, List

import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

_OCR_SUPPORTED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


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


_OCR_PROMPT = """あなたは財務諸表OCR専門AIです。添付の画像またはPDFから財務指標を読み取り、百万円単位のJSONのみを返してください。

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
  "other_assets": <その他固定資産合計>
}

重要ルール:
1. 赤字・損失は必ず負値で返す（▲500 → -500、△2,000 → -2000）。負値をゼロや正値に変換しない
2. 元の書類が「円」単位なら100万で除算、「千円」単位なら1000で除算して百万円に統一
3. 複数期のデータがある場合は最新期（当期）の値を使用
4. 値が不明・記載なしの場合はnullを返す（実際にゼロの場合のみ0を使う）
5. JSON以外の文字列は絶対に返さない"""


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
4. 代表者名はフルネームで文字列として返す。
5. 役員は、代表者以外の取締役、監査役などの役職を持つ人物名を文字列の配列として返す。役員が見つからない場合は空の配列を返す。
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
5. JSON以外の文字列は絶対に返さない。"""


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


@router.post("/ocr")
async def ocr_financial(file: UploadFile = File(...), doc_type: Optional[str] = None) -> JSONResponse:
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
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }

    try:
        resp = requests.post(
            _ocr_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        result = json.loads(text)

        # doc_type='tax_cert' の場合の追加検証
        if doc_type == "tax_cert":
            if not isinstance(result.get("tax_default"), bool):
                raise ValueError("tax_default must be a boolean.")
            if result.get("tax_amount") is not None and not isinstance(result.get("tax_amount"), int):
                raise ValueError("tax_amount must be an integer or null.")
        # doc_type='toukibo' の場合の追加検証
        elif doc_type == "toukibo":
            if not isinstance(result.get("established_date"), str):
                raise ValueError("established_date must be a string.")
            if result.get("capital") is not None and not isinstance(result.get("capital"), int):
                raise ValueError("capital must be an integer or null.")
            if not isinstance(result.get("representative"), str):
                raise ValueError("representative must be a string.")
            if not isinstance(result.get("directors"), list):
                raise ValueError("directors must be a list.")
            for director in result.get("directors", []):
                if not isinstance(director, str):
                    raise ValueError("Each director in 'directors' must be a string.")
        # doc_type='estimate' の場合の追加検証
        elif doc_type == "estimate":
            if not isinstance(result.get("item_name"), str):
                raise ValueError("item_name must be a string.")
            if result.get("amount") is not None and not isinstance(result.get("amount"), int):
                raise ValueError("amount must be an integer or null.")
            if not isinstance(result.get("maker"), str):
                raise ValueError("maker must be a string.")
            if not isinstance(result.get("model_number"), str):
                raise ValueError("model_number must be a string.")
        # doc_type='brochure' の場合の追加検証
        elif doc_type == "brochure":
            if not isinstance(result.get("industry"), str):
                raise ValueError("industry must be a string.")
            if not isinstance(result.get("business_description"), str):
                raise ValueError("business_description must be a string.")
            if not isinstance(result.get("main_clients"), list):
                raise ValueError("main_clients must be a list.")
            for client in result.get("main_clients", []):
                if not isinstance(client, str):
                    raise ValueError("Each client in 'main_clients' must be a string.")
            if result.get("employee_count") is not None and not isinstance(result.get("employee_count"), int):
                raise ValueError("employee_count must be an integer or null.")

        return JSONResponse(content=result)
    except requests.HTTPError as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini API エラー: {exc}"})
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini レスポンス解析エラーまたは検証エラー: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"OCR処理に失敗しました: {exc}"})

