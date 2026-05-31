"""
OCRモバイル入力コンポーネント

スマホカメラで撮影した決算書・審査資料の写真をアップロードし、
Gemini Vision API で財務データを自動抽出して審査フォームに入力する。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import streamlit as st


# 日本語キー → session_state キー（すべて千円単位）
_FIELD_MAP: dict[str, str] = {
    "売上高":      "nenshu",
    "売上総利益":  "item9_gross",
    "営業利益":    "rieki",
    "経常利益":    "item4_ord_profit",
    "当期純利益":  "item5_net_income",
    "純資産":      "net_assets",
    "総資産":      "total_assets",
    "減価償却費":  "item10_dep",
    "賃借料":      "item8_rent",
    "銀行借入":    "bank_credit",
    "リース残高":  "lease_credit",
}

_LABEL_MAP: dict[str, str] = {
    "nenshu":           "売上高",
    "item9_gross":      "売上総利益",
    "rieki":            "営業利益",
    "item4_ord_profit": "経常利益",
    "item5_net_income": "当期純利益",
    "net_assets":       "純資産",
    "total_assets":     "総資産",
    "item10_dep":       "減価償却費",
    "item8_rent":       "賃借料",
    "bank_credit":      "銀行借入",
    "lease_credit":     "リース残高",
}

_GEMINI_PROMPT = """
あなたはリース審査の財務データ抽出専門家です。
この画像は企業の決算書・財務資料・審査申込書などです。

以下の財務項目を画像から読み取り、JSON形式のみで返してください（説明文不要）。

金額は必ず**千円単位の整数**に変換してください:
- 万円表記 → ×10（例: 1500万円 → 15000）
- 百万円表記 → ×1000（例: 150百万円 → 150000）
- 億円表記 → ×100000（例: 1億円 → 100000）

{
  "企業名": "文字列（読み取れた場合のみ、なければnull）",
  "売上高": 千円単位の整数またはnull,
  "売上総利益": 千円単位の整数またはnull,
  "営業利益": 千円単位の整数またはnull,
  "経常利益": 千円単位の整数またはnull,
  "当期純利益": 千円単位の整数またはnull,
  "純資産": 千円単位の整数またはnull,
  "総資産": 千円単位の整数またはnull,
  "減価償却費": 千円単位の整数またはnull,
  "賃借料": 千円単位の整数またはnull,
  "銀行借入": 千円単位の整数またはnull,
  "リース残高": 千円単位の整数またはnull
}

注意: 数値はカンマ・通貨記号を除いた整数のみ。読み取れない項目は必ずnull。
"""


def _get_gemini_key() -> str:
    return (
        st.session_state.get("gemini_api_key", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


def _resize_image_bytes(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """4MB API制限対策として最大3000pxにリサイズし JPEG で返す。"""
    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes))
        max_side = 3000
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, PILImage.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime_type


def _ocr_with_gemini(image_bytes: bytes, mime_type: str, api_key: str) -> dict[str, Any]:
    """Gemini Vision REST API で画像を解析して財務データ辞書を返す。"""
    import requests  # type: ignore

    image_bytes, mime_type = _resize_image_bytes(image_bytes, mime_type)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    model = "gemini-1.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": _GEMINI_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": image_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }

    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini APIエラー {resp.status_code}: {resp.text[:200]}")

    raw_text = (
        resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    )

    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"JSONを抽出できませんでした:\n{raw_text[:300]}")

    return json.loads(json_match.group())


def _apply_ocr_result(result: dict[str, Any]) -> list[str]:
    """OCR結果を session_state に反映し、設定された項目名リストを返す。"""
    applied: list[str] = []

    # 企業名（文字列・最大100文字・英数字・日本語のみ許可）
    raw_name = result.get("企業名")
    if raw_name and isinstance(raw_name, str):
        company_name = raw_name.strip()[:100]
        if company_name:
            st.session_state["company_name"] = company_name
            applied.append(f"企業名: {company_name}")

    # 数値フィールド
    for jp_key, ss_key in _FIELD_MAP.items():
        val = result.get(jp_key)
        if val is None:
            continue
        try:
            # カンマや全角カンマを除去してから変換
            cleaned = str(val).replace(",", "").replace("，", "")
            int_val = int(float(cleaned))
        except (ValueError, TypeError):
            continue
        # 異常値ガード（0以上・9兆千円未満）
        if int_val < 0 or int_val > 9_000_000_000:
            continue
        st.session_state[ss_key] = int_val
        label = _LABEL_MAP.get(ss_key, ss_key)
        applied.append(f"{label}: {int_val:,}千円")

    return applied


def render_ocr_mobile_input() -> None:
    """OCRモバイル入力UIを描画する。form_apply の expander として使用。"""
    with st.expander("📷 OCR入力（スマホカメラ・書類写真から自動入力）", expanded=False):
        st.caption(
            "決算書・損益計算書・貸借対照表などの写真をアップロードすると、"
            "Gemini Vision AI が財務データを読み取り審査フォームへ自動入力します。"
        )

        api_key = _get_gemini_key()
        if not api_key:
            st.warning(
                "⚠️ Gemini APIキーが未設定です。\n"
                "サイドバーでAPIキーを入力するか、`GEMINI_API_KEY` 環境変数を設定してください。"
            )
            return

        uploaded = st.file_uploader(
            "書類写真を選択（スマホはカメラ撮影も可）",
            type=["jpg", "jpeg", "png", "webp", "heic"],
            key="ocr_upload",
            help="スマホカメラで撮影した決算書・損益計算書・貸借対照表などに対応しています。",
        )

        if uploaded is None:
            return

        st.image(uploaded, caption="アップロードした書類", use_container_width=True)

        if st.button("🔍 OCR実行（財務データを自動抽出）", key="ocr_run_btn"):
            with st.spinner("Gemini Vision AI で書類を解析中..."):
                try:
                    image_bytes = uploaded.read()
                    suffix = (
                        uploaded.name.rsplit(".", 1)[-1].lower()
                        if "." in uploaded.name
                        else "jpg"
                    )
                    mime_map = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "webp": "image/webp",
                        "heic": "image/heic",
                    }
                    mime_type = mime_map.get(suffix, "image/jpeg")

                    result = _ocr_with_gemini(image_bytes, mime_type, api_key)
                    st.session_state["_ocr_last_result"] = result

                except Exception as exc:
                    st.error(f"OCR処理中にエラーが発生しました: {exc}")
                    return

        ocr_result = st.session_state.get("_ocr_last_result")
        if not ocr_result:
            return

        st.markdown("#### 📋 抽出された財務データ")
        # 抽出結果を読みやすい表形式で表示
        rows = []
        name_val = ocr_result.get("企業名")
        if name_val:
            rows.append({"項目": "企業名", "抽出値": str(name_val)[:100]})
        for jp_key in _FIELD_MAP:
            val = ocr_result.get(jp_key)
            if val is not None:
                try:
                    int_val = int(float(str(val).replace(",", "").replace("，", "")))
                    rows.append({"項目": jp_key, "抽出値": f"{int_val:,}千円"})
                except (ValueError, TypeError):
                    pass

        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("財務データを読み取れませんでした。画質・角度を確認して再撮影してください。")

        col_apply, col_clear = st.columns([2, 1])
        with col_apply:
            if st.button("✅ フォームに入力する", key="ocr_apply_btn", type="primary"):
                applied = _apply_ocr_result(ocr_result)
                if applied:
                    st.success(
                        f"✅ {len(applied)} 項目を自動入力しました:\n"
                        + "\n".join(f"  • {x}" for x in applied)
                    )
                    del st.session_state["_ocr_last_result"]
                    st.rerun()
                else:
                    st.warning("反映できる財務データがありませんでした。手入力してください。")
        with col_clear:
            if st.button("🗑️ クリア", key="ocr_clear_btn"):
                if "_ocr_last_result" in st.session_state:
                    del st.session_state["_ocr_last_result"]
                st.rerun()
