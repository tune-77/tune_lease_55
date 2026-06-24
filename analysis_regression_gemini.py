"""
analysis_regression_gemini.py
==============================
analysis_regression.py と同じシグネチャ・戻り値 dict を返すが、
LightGBM/ロジスティック回帰の代わりに Gemini API で計算する。

Gemini に CSV 形式のデータを送り、同じキー構造の JSON を返してもらう。
"""
from __future__ import annotations
import json
import re
import streamlit as st
from data_cases import load_all_cases
from secret_manager import get_gemini_api_key
from config import GEMINI_MODEL_DEFAULT
from analysis_regression import (
    QUALITATIVE_ANALYSIS_MIN_CASES,
    INDUSTRY_BASES,
    BENCH_BASES,
    COEFF_MAIN_KEYS,
    COEFF_EXTRA_KEYS,
    COEFF_LABELS,
    build_design_matrix_from_logs,
    build_design_matrix_from_logs_by_industry,
)

# ── Gemini 呼び出し共通 ──────────────────────────────────────────────────────

def _gemini_call(prompt: str, max_tokens: int = 4000) -> str | None:
    api_key = (
        st.session_state.get("gemini_api_key", "").strip()
        or get_gemini_api_key()
        or ""
    )
    if not api_key:
        return None
    model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT) or "gemini-2.0-flash"
    if "1.5" in model:
        model = "gemini-2.0-flash"

    try:
        import google.genai as _genai
        from google.genai import types as _types
        client = _genai.Client(api_key=api_key)
        cfg = _types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.1)
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        text = None
        try:
            text = resp.text
        except Exception:
            pass
        if not text and getattr(resp, "candidates", None):
            for cand in resp.candidates:
                if getattr(cand, "content", None):
                    for part in (cand.content.parts or []):
                        if getattr(part, "text", None):
                            text = (text or "") + part.text
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        import google.generativeai as _old
        _old.configure(api_key=api_key)
        m = _old.GenerativeModel(
            model_name=model,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1},
        )
        resp = m.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return None


def _extract_json(text: str) -> dict | None:
    """レスポンステキストから JSON ブロックを抽出してパースする。"""
    # ```json ... ``` ブロックを優先
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        # 最初の { ... } を探す
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = m.group(1) if m else text
    try:
        return json.loads(raw)
    except Exception:
        return None


def _build_analysis_prompt(
    feature_names: list[str],
    rows: list[list],
    y_list: list[int],
    label_map: dict[str, str] | None = None,
    mode: str = "qualitative",
) -> str:
    """特徴量行列 + 目的変数からGemini分析プロンプトを生成する。"""
    n = len(y_list)
    n_pos = sum(y_list)
    n_neg = n - n_pos

    # CSV ヘッダー + データ（最大 200 件まで）
    header = "result," + ",".join(feature_names)
    lines = [header]
    for i, (row, y) in enumerate(zip(rows, y_list)):
        if i >= 200:
            break
        label = "成約" if y == 1 else "失注"
        vals = ",".join(
            str(round(v, 3)) if isinstance(v, float) else str(v) for v in row
        )
        lines.append(f"{label},{vals}")
    csv_text = "\n".join(lines)

    # 特徴量の日本語ラベル
    feat_labels = [label_map.get(f, f) if label_map else f for f in feature_names]
    feat_list = "\n".join(f"- {fn}: {lbl}" for fn, lbl in zip(feature_names, feat_labels))

    prompt = f"""あなたはリース審査の統計分析の専門家です。
以下のデータはTune式リース審査システムの成約・失注案件です（成約=1, 失注=0）。
総件数: {n}件（成約{n_pos}件 / 失注{n_neg}件）

## 特徴量説明
{feat_list}

## データ (CSV形式)
{csv_text}

## 依頼
上記データを分析し、以下のキーを持つ JSON を **コードブロックなしで** そのまま返してください。
数値はすべて float 型で返してください。

{{
  "lr_coef": [["特徴量名", 係数], ...],
  "lr_intercept": 切片値,
  "accuracy_lr": ロジスティック回帰の推定正解率(0-1),
  "auc_lr": ロジスティック回帰の推定AUC(0-1),
  "lgb_importance": [["特徴量名", 重要度スコア], ...],
  "accuracy_lgb": LightGBM相当の推定正解率(0-1),
  "auc_lgb": LightGBM相当の推定AUC(0-1),
  "ensemble_alpha": LRとLGBの最適ブレンド比率alpha(0-1),
  "auc_ensemble": アンサンブルの推定AUC(0-1),
  "accuracy_ensemble": アンサンブルの推定正解率(0-1)
}}

ルール：
- lr_coef: 全 {len(feature_names)} 特徴量を含める。正値=成約に有利、負値=失注に有利（-2〜2の範囲）
- lgb_importance: 全 {len(feature_names)} 特徴量を含める。0〜100の整数スコアで重要度順に並べる
- 推定正解率・AUCはデータの成約率と特徴量の分離度から推定する（0.50〜0.99の範囲）
- JSON のみ返す（説明文不要）
"""
    return prompt


def _parse_result(
    raw: dict,
    feature_names: list[str],
    n_cases: int,
    n_pos: int,
    n_neg: int,
) -> dict:
    """Gemini の JSON レスポンスを analysis_regression と同じ dict 構造に変換する。"""
    out: dict = {
        "n_cases": n_cases,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "feature_names": feature_names,
        "shap_importance": [],
    }

    def _to_pairs(lst, feature_names):
        result = []
        fn_set = {f: i for i, f in enumerate(feature_names)}
        for item in lst:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                fname, val = item
                try:
                    result.append((str(fname), float(val)))
                except Exception:
                    pass
        # 欠落している特徴量は 0 で補完
        found = {k for k, _ in result}
        for f in feature_names:
            if f not in found:
                result.append((f, 0.0))
        return result

    if "lr_coef" in raw:
        out["lr_coef"] = _to_pairs(raw["lr_coef"], feature_names)
    if "lr_intercept" in raw:
        try:
            out["lr_intercept"] = float(raw["lr_intercept"])
        except Exception:
            out["lr_intercept"] = 0.0
    for key in ("accuracy_lr", "auc_lr", "accuracy_lgb", "auc_lgb",
                "ensemble_alpha", "auc_ensemble", "accuracy_ensemble"):
        if key in raw:
            try:
                out[key] = float(raw[key])
            except Exception:
                pass
    if "lgb_importance" in raw:
        out["lgb_importance"] = _to_pairs(raw["lgb_importance"], feature_names)

    return out


# ── 定性要因分析 ────────────────────────────────────────────────────────────

def run_qualitative_contract_analysis_gemini(qual_correction_items) -> dict | None:
    """
    定性項目のみで Gemini に成約/失注分析をさせる。
    run_qualitative_contract_analysis() と同じ戻り値構造。
    """
    if not qual_correction_items:
        return None
    cases = load_all_cases()
    registered = [c for c in cases if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < 3:
        return None

    qual_ids = [it["id"] for it in qual_correction_items]
    feature_names = [
        "取引区分_メイン先",
        "競合状況_競合あり",
        "顧客区分_新規先",
        "商談ソース_銀行紹介",
        "リース物件",
    ] + [it["label"] for it in qual_correction_items]

    rows, y_list = [], []
    asset_ids = list({
        (c.get("inputs") or {}).get("lease_asset_id") or
        (c.get("inputs") or {}).get("lease_asset_name") or "未選択"
        for c in registered
    })
    asset_to_idx = {a: i for i, a in enumerate(asset_ids)}

    for c in registered:
        inp = c.get("inputs") or {}
        y_list.append(1 if c.get("final_status") == "成約" else 0)
        main_bank = c.get("main_bank") or inp.get("main_bank") or "非メイン先"
        competitor = c.get("competitor") or inp.get("competitor") or "競合なし"
        customer_type = c.get("customer_type") or inp.get("customer_type") or "既存先"
        deal_source = inp.get("deal_source") or "その他"
        asset_id = inp.get("lease_asset_id") or inp.get("lease_asset_name") or "未選択"
        row = [
            1.0 if main_bank == "メイン先" else 0.0,
            1.0 if competitor == "競合あり" else 0.0,
            1.0 if customer_type == "新規先" else 0.0,
            1.0 if deal_source == "銀行紹介" else 0.0,
            float(asset_to_idx.get(asset_id, 0)),
        ]
        q = (c.get("result") or {}).get("qualitative_scoring_correction") or inp.get("qualitative_scoring") or {}
        items_data = q.get("items") or {}
        for it in qual_correction_items:
            val = items_data.get(it["id"], {})
            v = val.get("value") if isinstance(val, dict) else None
            row.append(float(v) if isinstance(v, (int, float)) else -1.0)
        rows.append(row)

    n = len(y_list)
    n_pos = sum(y_list)
    n_neg = n - n_pos

    prompt = _build_analysis_prompt(feature_names, rows, y_list, mode="qualitative")
    text = _gemini_call(prompt, max_tokens=3000)
    if not text:
        return None

    raw = _extract_json(text)
    if not raw:
        return None

    return _parse_result(raw, feature_names, n, n_pos, n_neg)


# ── 定量要因分析 ────────────────────────────────────────────────────────────

def run_quantitative_contract_analysis_gemini() -> dict | None:
    """
    定量項目で Gemini に成約/失注分析をさせる。
    run_quantitative_contract_analysis() と同じ戻り値構造。
    """
    import numpy as np
    all_logs = load_all_cases()
    X, y = build_design_matrix_from_logs(all_logs, model_key=None)
    if X is None or y is None or len(y) < 3:
        return None

    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    rows = X.tolist()
    y_list = y.tolist()

    prompt = _build_analysis_prompt(
        feature_names, rows, y_list,
        label_map=COEFF_LABELS,
        mode="quantitative",
    )
    text = _gemini_call(prompt, max_tokens=4000)
    if not text:
        return None

    raw = _extract_json(text)
    if not raw:
        return None

    return _parse_result(raw, feature_names, n, n_pos, n_neg)


# ── 業種ごと定量分析 ─────────────────────────────────────────────────────────

def run_quantitative_by_industry_gemini() -> dict | None:
    """
    業種ベースごとに Gemini で定量分析。
    run_quantitative_by_industry() と同じ戻り値構造。
    """
    import numpy as np
    all_logs = load_all_cases()
    registered = [c for c in all_logs if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < 5:
        return None

    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    results = {}
    for base in INDUSTRY_BASES:
        X, y = build_design_matrix_from_logs_by_industry(all_logs, base)
        if X is None or len(y) < 3:
            results[base] = {"skip": True, "reason": "データなしまたは3件未満"}
            continue
        n_orig = len(y)
        bootstrapped = False
        # 件数不足時はブートストラップ
        if len(y) < QUALITATIVE_ANALYSIS_MIN_CASES:
            idx = np.random.choice(len(y), QUALITATIVE_ANALYSIS_MIN_CASES, replace=True)
            X = X[idx]
            y = y[idx]
            bootstrapped = True
        rows = X.tolist()
        y_list = y.tolist()
        prompt = _build_analysis_prompt(
            feature_names, rows, y_list,
            label_map=COEFF_LABELS,
            mode="quantitative",
        )
        text = _gemini_call(prompt, max_tokens=3000)
        if not text:
            results[base] = {"skip": True, "reason": "Gemini API エラー"}
            continue
        raw = _extract_json(text)
        if not raw:
            results[base] = {"skip": True, "reason": "JSON パース失敗"}
            continue
        r = _parse_result(raw, feature_names, len(y_list), sum(y_list), len(y_list) - sum(y_list))
        r["n_cases_orig"] = n_orig
        r["bootstrapped"] = bootstrapped
        results[base] = r
    return results


# ── 指標ごと定量分析 ─────────────────────────────────────────────────────────

def run_quantitative_by_indicator_gemini() -> dict | None:
    """
    指標ベースごとに Gemini で定量分析。
    run_quantitative_by_indicator() と同じ戻り値構造。
    """
    from analysis_regression import (
        build_design_matrix_indicator_by_bench,
        INDICATOR_MAIN_KEYS,
    )
    import numpy as np
    all_logs = load_all_cases()
    registered = [c for c in all_logs if c.get("final_status") in ["成約", "失注"]]
    if len(registered) < 5:
        return None

    ind_extra = [k for k in COEFF_EXTRA_KEYS if k != "qualitative_combined"][:8]
    feature_names = INDICATOR_MAIN_KEYS + ind_extra
    results = {}
    for bench in BENCH_BASES:
        try:
            X, y = build_design_matrix_indicator_by_bench(all_logs, bench)
        except Exception as e:
            results[bench] = {"skip": True, "reason": str(e)}
            continue
        if X is None or len(y) < 3:
            results[bench] = {"skip": True, "reason": "データなしまたは3件未満"}
            continue
        n_orig = len(y)
        bootstrapped = False
        if len(y) < QUALITATIVE_ANALYSIS_MIN_CASES:
            idx = np.random.choice(len(y), QUALITATIVE_ANALYSIS_MIN_CASES, replace=True)
            X = X[idx]
            y = y[idx]
            bootstrapped = True
        rows = X.tolist()
        y_list = y.tolist()
        fn = feature_names[:X.shape[1]] if X.shape[1] < len(feature_names) else feature_names
        prompt = _build_analysis_prompt(
            fn, rows, y_list,
            label_map=COEFF_LABELS,
            mode="quantitative",
        )
        text = _gemini_call(prompt, max_tokens=3000)
        if not text:
            results[bench] = {"skip": True, "reason": "Gemini API エラー"}
            continue
        raw = _extract_json(text)
        if not raw:
            results[bench] = {"skip": True, "reason": "JSON パース失敗"}
            continue
        r = _parse_result(raw, fn, len(y_list), sum(y_list), len(y_list) - sum(y_list))
        r["n_cases_orig"] = n_orig
        r["bootstrapped"] = bootstrapped
        r["feature_names"] = fn
        results[bench] = r
    return results
