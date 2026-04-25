"""案件編集 — 過去案件の定性評価・営業部・ステータスを後から補完・上書きする"""

import copy
import json
import sqlite3
from contextlib import closing

import pandas as pd
import streamlit as st

from constants import QUALITATIVE_SCORE_RANKS, QUALITATIVE_SCORING_CORRECTION_ITEMS
from data_cases import DB_PATH, get_score_weights, load_all_cases
from components.form_apply import SALES_DEPT_OPTIONS

_FINAL_STATUS_OPTIONS = ["未登録", "成約", "失注", "承認待ち", "否決", "審議中"]


def _calc_qual_correction(
    item_values: dict, final_score: float, w_quant: float, w_qual: float
) -> dict | None:
    qual_weight_sum = 0
    qual_weighted_total = 0.0
    qual_correction_items: dict = {}

    for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
        val = item_values.get(item["id"])
        if val is None:
            continue
        level_label = next((lbl for v, lbl in item["options"] if v == val), str(val))
        qual_correction_items[item["id"]] = {
            "value": val,
            "label": item["label"],
            "weight": item["weight"],
            "level_label": level_label,
        }
        qual_weight_sum += item["weight"]
        qual_weighted_total += (val / 4.0) * 100 * (item["weight"] / 100.0)

    if qual_weight_sum == 0:
        return None

    qual_weighted_score = round(qual_weighted_total / qual_weight_sum * 100)
    qual_weighted_score = min(100, max(0, qual_weighted_score))
    combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
    combined_score = min(100, max(0, combined_score))
    qual_rank = next(
        (r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]),
        QUALITATIVE_SCORE_RANKS[-1],
    )
    return {
        "items": qual_correction_items,
        "weighted_score": qual_weighted_score,
        "combined_score": combined_score,
        "rank": qual_rank["label"],
        "rank_text": qual_rank["text"],
        "rank_desc": qual_rank["desc"],
    }


def _update_case_full(case_id: str, merged_data: dict, final_status: str, sales_dept: str) -> bool:
    """case_id のレコードを data / final_status / sales_dept 列ごと更新する。"""
    if not case_id or not DB_PATH:
        return False
    try:
        from data_cases import CustomJSONEncoder  # type: ignore[attr-defined]
        json_str = json.dumps(merged_data, ensure_ascii=False, cls=CustomJSONEncoder)
    except Exception:
        json_str = json.dumps(merged_data, ensure_ascii=False)
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "UPDATE past_cases SET data = ?, final_status = ?, sales_dept = ? WHERE id = ?",
                (json_str, final_status, sales_dept, case_id),
            )
            conn.commit()
        return True
    except Exception as exc:
        import sys
        print(f"[case_editor] update error: {exc}", file=sys.stderr)
        return False


def _get_display_name(case: dict) -> str:
    inp = case.get("inputs", {})
    return (
        case.get("company_name")
        or inp.get("company_name")
        or case.get("borrower_name")
        or ""
    )


def _get_company_no(case: dict) -> str:
    inp = case.get("inputs", {})
    return case.get("company_no") or inp.get("company_no") or ""


def render_case_editor() -> None:
    st.title("📝 案件修正")
    st.caption("過去案件の定性評価・営業部・ステータスを補完・上書きできます。財務数値は変更できません。")

    cases = load_all_cases()
    if not cases:
        st.warning("過去案件がありません。")
        return

    # ──────────────────────────────────────────
    # ① 案件一覧テーブル
    # ──────────────────────────────────────────
    rows = []
    for c in reversed(cases):
        inp = c.get("inputs", {})
        result = c.get("result", {})
        qual_scoring = inp.get("qualitative_scoring") or result.get("qualitative_scoring_correction")
        rows.append(
            {
                "_id": c.get("id", ""),
                "企業名": _get_display_name(c),
                "企業番号": _get_company_no(c),
                "日付": (c.get("timestamp", "") or "")[:10],
                "スコア": result.get("score", ""),
                "定性評価": "✅ あり" if qual_scoring else "🔴 未入力",
                "営業部": c.get("sales_dept", "未設定") or "未設定",
            }
        )

    df_display = pd.DataFrame(rows).drop(columns=["_id"])

    def _highlight(row: pd.Series):
        if row["定性評価"] == "🔴 未入力":
            return ["background-color: #fff0f0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(_highlight, axis=1),
        use_container_width=True,
        height=300,
    )

    # ──────────────────────────────────────────
    # 案件選択
    # ──────────────────────────────────────────
    case_labels = [
        f"{r['日付']} | {r['企業名'] or '(無名)'} | スコア:{r['スコア']} | {r['定性評価']}"
        for r in rows
    ]
    selected_idx = st.selectbox(
        "編集する案件を選択",
        range(len(case_labels)),
        format_func=lambda i: case_labels[i],
        key="case_editor_select",
    )
    if selected_idx is None:
        return

    case_id = rows[selected_idx]["_id"]
    case = next((c for c in cases if c.get("id") == case_id), None)
    if not case:
        st.error("案件データを取得できませんでした。")
        return

    inp = case.get("inputs", {})
    result = case.get("result", {})
    qualitative = inp.get("qualitative", {})
    existing_qual_scoring = inp.get("qualitative_scoring") or result.get("qualitative_scoring_correction") or {}

    st.divider()
    st.subheader(f"✏️ 編集: {_get_display_name(case) or '(無名)'}")

    # ──────────────────────────────────────────
    # ② 編集フォーム
    # ──────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # 営業部
        current_dept = case.get("sales_dept", "未設定") or "未設定"
        dept_idx = SALES_DEPT_OPTIONS.index(current_dept) if current_dept in SALES_DEPT_OPTIONS else 0
        new_dept = st.selectbox("営業部", SALES_DEPT_OPTIONS, index=dept_idx, key=f"edit_dept_{case_id}")

    with col_right:
        # 最終ステータス
        current_status = case.get("final_status", "未登録") or "未登録"
        status_idx = (
            _FINAL_STATUS_OPTIONS.index(current_status)
            if current_status in _FINAL_STATUS_OPTIONS
            else 0
        )
        new_status = st.selectbox(
            "最終ステータス",
            _FINAL_STATUS_OPTIONS,
            index=status_idx,
            key=f"edit_status_{case_id}",
        )

    # 財務数値（参照のみ）
    with st.expander("📊 財務数値（参照のみ・編集不可）", expanded=False):
        fin = result.get("financials", {})
        nenshu = inp.get("nenshu") or fin.get("nenshu")
        op = inp.get("op_profit") or fin.get("op_profit")
        eq = result.get("user_eq")
        c1, c2, c3 = st.columns(3)
        c1.metric("売上高（千円）", f"{int(nenshu):,}" if isinstance(nenshu, (int, float)) else "—")
        c2.metric("営業利益（千円）", f"{int(op):,}" if isinstance(op, (int, float)) else "—")
        c3.metric("自己資本比率", f"{eq:.1%}" if isinstance(eq, float) else "—")

    st.subheader("📋 定性スコアリング")

    # 定性フリーテキスト
    current_passion = qualitative.get("passion_text", "") or ""
    new_passion = st.text_area(
        "定性評価フリーテキスト（熱意・裏事情など）",
        value=current_passion,
        key=f"edit_passion_{case_id}",
        height=80,
    )

    # 定性スコアリング各項目
    qual_vals: dict = {}
    item_cols = st.columns(2)
    for i, item in enumerate(QUALITATIVE_SCORING_CORRECTION_ITEMS):
        existing_item = (existing_qual_scoring.get("items") or {}).get(item["id"], {})
        existing_val = existing_item.get("value")
        opts_none = [(None, "（未入力）")]
        all_opts = opts_none + list(item["options"])
        opt_values = [v for v, _ in all_opts]
        opt_labels = [lbl for _, lbl in all_opts]
        current_idx = opt_values.index(existing_val) if existing_val in opt_values else 0
        with item_cols[i % 2]:
            chosen_label = st.selectbox(
                f"{item['label']}（重み {item['weight']}%）",
                opt_labels,
                index=current_idx,
                key=f"edit_qual_{case_id}_{item['id']}",
            )
            qual_vals[item["id"]] = opt_values[opt_labels.index(chosen_label)]

    # ──────────────────────────────────────────
    # ③ 保存処理
    # ──────────────────────────────────────────
    if st.button("💾 保存する", type="primary", key=f"save_case_{case_id}"):
        _, _, w_quant, w_qual = get_score_weights()
        final_score = float(result.get("score") or 50)

        new_qual_correction = _calc_qual_correction(qual_vals, final_score, w_quant, w_qual)

        new_case = copy.deepcopy(case)

        # inputs 更新
        new_case.setdefault("inputs", {}).setdefault("qualitative", {})
        new_case["inputs"]["qualitative"]["passion_text"] = new_passion
        new_case["inputs"]["qualitative_scoring"] = new_qual_correction

        # result 更新
        new_case.setdefault("result", {})
        new_case["result"]["qualitative_scoring_correction"] = new_qual_correction
        if new_qual_correction:
            new_case["result"]["hantei_score"] = new_qual_correction["combined_score"]

        # top-level フィールド更新
        new_case["sales_dept"] = new_dept
        new_case["final_status"] = new_status

        ok = _update_case_full(case_id, new_case, new_status, new_dept)
        if ok:
            st.success("✅ 保存しました。")
            try:
                from data_cases import load_all_cases_cached
                load_all_cases_cached.clear()
            except Exception:
                pass
            st.rerun()
        else:
            st.error("保存に失敗しました。DBパスを確認してください。")
