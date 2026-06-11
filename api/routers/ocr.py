import base64
import json
import os

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


@router.post("/ocr")
async def ocr_financial(file: UploadFile = File(...)) -> JSONResponse:
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
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": file.content_type, "data": file_b64}},
                    {"text": _OCR_PROMPT},
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
        return JSONResponse(content=result)
    except requests.HTTPError as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini API エラー: {exc}"})
    except (KeyError, json.JSONDecodeError) as exc:
        return JSONResponse(status_code=502, content={"error": f"Gemini レスポンス解析エラー: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"OCR処理に失敗しました: {exc}"})
