import datetime
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ai_chat import _gemini_chat
from data_cases import load_all_cases
from analysis_regression import run_contract_driver_analysis
from industry_normalizer import normalize_industry_major, normalize_industry_sub
from secret_manager import get_gemini_api_key
from config import GEMINI_MODEL_DEFAULT
from report_pdf import build_contract_report_pdf

SALES_DEPT_OPTIONS = ["宇都宮営業部", "小山営業部", "足利営業部", "埼玉営業部"]
MONTH_FILTER_START = "2025-04"
STAT_TEST_MIN_CASES = 8
STAT_TEST_PERMUTATIONS = 500


def _normalize_sales_dept(value: object) -> str:
    dept = str(value or "").strip()
    if dept in ("", "0", "未設定", "未読取"):
        return ""
    return dept if dept in SALES_DEPT_OPTIONS else ""


def _display_sales_dept(value: object) -> str:
    return _normalize_sales_dept(value) or "未読取"


def _safe_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        num = float(value)
        if num <= 0:
            return None
        if num > 1000:
            num = num / 1000.0
        elif num > 100:
            num = num / 100.0
        return num
    except (TypeError, ValueError):
        return None


def _case_timestamp(case: dict) -> str:
    for key in ("timestamp", "審査日", "registration_date", "final_result_date"):
        raw = str(case.get(key) or "").strip()
        if raw:
            return raw
    return ""


def _case_month(case: dict) -> str:
    raw = _case_timestamp(case)
    if not raw:
        return ""
    if len(raw) >= 7 and raw[4] == "-":
        return raw[:7]
    if len(raw) >= 7 and raw[4] == "/":
        parts = raw.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}-{int(parts[1]):02d}"
    return raw[:7]


def _case_major(case: dict) -> str:
    major = (
        case.get("industry_major")
        or case.get("result", {}).get("industry_major")
        or case.get("inputs", {}).get("industry_major")
        or ""
    )
    return normalize_industry_major(major) or str(major or "不明")


def _case_sub(case: dict) -> str:
    major = _case_major(case)
    sub = (
        case.get("industry_sub")
        or case.get("result", {}).get("industry_sub")
        or case.get("inputs", {}).get("industry_sub")
        or ""
    )
    return normalize_industry_sub(sub, major) or str(sub or "不明")


def _case_final_rate(case: dict) -> float | None:
    rate = (
        case.get("final_rate")
        or case.get("result", {}).get("final_rate")
        or case.get("inputs", {}).get("final_rate")
    )
    return _safe_float(rate)


def _case_metric_value(case: dict, metric_key: str) -> float | None:
    result = case.get("result", {}) or {}
    inputs = case.get("inputs", {}) or {}
    financials = case.get("financials", {}) or {}

    source_map = {
        "final_rate": case.get("final_rate") or result.get("final_rate") or inputs.get("final_rate"),
        "score": case.get("score") or result.get("score"),
        "user_eq": case.get("user_eq") or result.get("user_eq"),
        "op_profit": financials.get("op_profit") or inputs.get("op_profit") or result.get("op_profit"),
        "ord_profit": financials.get("ord_profit") or inputs.get("ord_profit") or result.get("ord_profit"),
        "net_income": financials.get("net_income") or inputs.get("net_income") or result.get("net_income"),
        "gross_profit": financials.get("gross_profit") or inputs.get("gross_profit") or result.get("gross_profit"),
        "bank_credit": financials.get("bank_credit") or inputs.get("bank_credit") or result.get("bank_credit"),
        "lease_credit": financials.get("lease_credit") or inputs.get("lease_credit") or result.get("lease_credit"),
        "acquisition_cost": inputs.get("acquisition_cost") or result.get("acquisition_cost"),
        "lease_term": inputs.get("lease_term") or result.get("lease_term"),
        "contracts": inputs.get("contracts") or result.get("contracts"),
    }
    return _safe_float(source_map.get(metric_key))


def _case_revenue(case: dict) -> float | None:
    result = case.get("result", {}) or {}
    inputs = case.get("inputs", {}) or {}
    financials = case.get("financials", {}) or {}
    return _safe_float(
        financials.get("nenshu")
        or inputs.get("nenshu")
        or result.get("nenshu")
        or case.get("nenshu")
    )


STAT_TEST_METRICS = [
    ("final_rate", "獲得レート(%)"),
    ("score", "審査スコア"),
    ("user_eq", "自己資本比率(%)"),
    ("op_profit", "営業利益(百万円)"),
    ("ord_profit", "経常利益(百万円)"),
    ("net_income", "当期純利益(百万円)"),
    ("gross_profit", "売上総利益(百万円)"),
    ("bank_credit", "銀行与信(百万円)"),
    ("lease_credit", "リース与信(百万円)"),
    ("acquisition_cost", "取得価格(百万円)"),
    ("lease_term", "リース期間(月)"),
]


def _build_dept_industry_frame(all_cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in all_cases:
        dept = _normalize_sales_dept(case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or "")
        if not dept:
            continue
        status = case.get("final_status") or "未登録"
        rows.append(
            {
                "営業部": dept,
                "業種大分類": _case_major(case),
                "業種小分類": _case_sub(case),
                "結果": status,
                "成約フラグ": 1 if status == "成約" else 0,
                "金利": _case_final_rate(case),
                "月": _case_month(case),
            }
        )
    return pd.DataFrame(rows)


def _build_monthly_frame(all_cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in all_cases:
        status = case.get("final_status") or "未登録"
        month = _case_month(case)
        if not month:
            continue
        if month < MONTH_FILTER_START:
            continue
        rows.append(
            {
                "月": month,
                "結果": status,
                "金利": _case_final_rate(case),
                "営業部": _normalize_sales_dept(case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or ""),
                "成約フラグ": 1 if status == "成約" else 0,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["営業部"].astype(str).isin(SALES_DEPT_OPTIONS)]
    df = df[df["結果"].isin(["成約", "失注"])]
    if df.empty:
        return df
    return df


def _build_stat_test_frame(all_cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in all_cases:
        month = _case_month(case)
        if not month or month < MONTH_FILTER_START:
            continue
        status = case.get("final_status") or "未登録"
        if status not in ("成約", "失注"):
            continue
        dept = _normalize_sales_dept(case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or "")
        if not dept:
            continue
        row = {"営業部": dept, "月": month, "結果": status}
        for key, _ in STAT_TEST_METRICS:
            row[key] = _case_metric_value(case, key)
        rows.append(row)
    return pd.DataFrame(rows)


def _anova_f_stat(groups: list[np.ndarray]) -> float | None:
    valid = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return None
    all_vals = np.concatenate(valid)
    overall_mean = float(np.mean(all_vals))
    ss_between = 0.0
    ss_within = 0.0
    total_n = 0
    for g in valid:
        n = len(g)
        mean = float(np.mean(g))
        ss_between += n * (mean - overall_mean) ** 2
        ss_within += float(np.sum((g - mean) ** 2))
        total_n += n
    df_between = len(valid) - 1
    df_within = total_n - len(valid)
    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return None
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return ms_between / ms_within if ms_within > 0 else None


def _permutation_p_value(values: np.ndarray, labels: np.ndarray, sizes: list[int], observed: float | None, n_perm: int = STAT_TEST_PERMUTATIONS) -> float | None:
    if observed is None or len(values) < sum(sizes) or len(sizes) < 2:
        return None
    rng = np.random.default_rng(42)
    count = 0
    total = 0
    for _ in range(n_perm):
        perm = rng.permutation(values)
        groups = []
        start = 0
        for size in sizes:
            groups.append(perm[start:start + size])
            start += size
        f = _anova_f_stat(groups)
        if f is None:
            continue
        total += 1
        if f >= observed:
            count += 1
    if total == 0:
        return None
    return (count + 1) / (total + 1)


def _run_stat_significance_analysis(df_stat: pd.DataFrame) -> pd.DataFrame:
    if df_stat.empty:
        return pd.DataFrame()
    rows = []
    for key, label in STAT_TEST_METRICS:
        sub = df_stat[["営業部", key]].dropna()
        if sub.empty:
            continue
        groups = [g[key].dropna().to_numpy(dtype=float) for _, g in sub.groupby("営業部")]
        groups = [g for g in groups if len(g) >= STAT_TEST_MIN_CASES]
        if len(groups) < 2:
            continue
        observed = _anova_f_stat(groups)
        if observed is None:
            continue
        values = np.concatenate(groups)
        sizes = [len(g) for g in groups]
        p_value = _permutation_p_value(values, np.array([0] * len(values)), sizes, observed)
        # 効果量 eta^2
        overall_mean = float(np.mean(values))
        ss_between = sum(len(g) * (float(np.mean(g)) - overall_mean) ** 2 for g in groups)
        ss_total = float(np.sum((values - overall_mean) ** 2))
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        group_stats = (
            sub.groupby("営業部")[key]
            .agg(["count", "mean", "median"])
            .rename(columns={"count": "件数", "mean": "平均", "median": "中央値"})
            .reset_index()
        )
        top = group_stats.sort_values("平均", ascending=False).iloc[0]
        bottom = group_stats.sort_values("平均", ascending=True).iloc[0]
        rows.append({
            "項目": label,
            "N": int(len(values)),
            "営業部数": int(len(groups)),
            "F値": float(observed),
            "p値": float(p_value) if p_value is not None else None,
            "eta^2": float(eta2),
            "最も高い営業部": str(top["営業部"]),
            "最も高い平均": float(top["平均"]),
            "最も低い営業部": str(bottom["営業部"]),
            "最も低い平均": float(bottom["平均"]),
            "有意": "有意" if p_value is not None and p_value < 0.05 else "非有意",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["有意", "p値", "eta^2"], ascending=[False, True, False])
    return out


@st.cache_data(ttl=600, show_spinner=False)
def _compute_department_stat_analysis(all_cases: list[dict]) -> dict:
    rows = []
    for case in all_cases:
        dept = _normalize_sales_dept(case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or "")
        if not dept:
            continue
        major = _case_major(case)
        revenue = _case_revenue(case)
        if revenue is None:
            continue
        rows.append(
            {
                "営業部": dept,
                "業種大分類": major,
                "売上高": revenue,
                "売上高_log": float(np.log1p(revenue)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        empty = pd.DataFrame()
        return {
            "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "analysis_df": empty,
            "industry_table": empty,
            "industry_summary": empty,
            "industry_top": empty,
            "revenue_summary": empty,
            "revenue_detail": empty,
            "stat_df": empty,
            "sig_summary": empty,
        }

    # 業種分布の検定
    industry_ct = pd.crosstab(df["営業部"], df["業種大分類"])
    industry_chi2 = None
    industry_p = None
    industry_dof = None
    industry_v = None
    industry_top = pd.DataFrame()
    if industry_ct.shape[0] >= 2 and industry_ct.shape[1] >= 2:
        chi2, p_value, dof, expected = stats.chi2_contingency(industry_ct)
        industry_chi2 = float(chi2)
        industry_p = float(p_value)
        industry_dof = int(dof)
        total_n = float(industry_ct.to_numpy().sum())
        industry_v = float(np.sqrt(chi2 / (total_n * (min(industry_ct.shape) - 1)))) if total_n > 0 else None

        expected_df = pd.DataFrame(expected, index=industry_ct.index, columns=industry_ct.columns)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = (industry_ct - expected_df) / np.sqrt(expected_df)
        industry_top = (
            residual.stack()
            .rename("残差")
            .reset_index()
            .rename(columns={"level_0": "営業部", "level_1": "業種大分類"})
        )
        industry_top["観測値"] = industry_top.apply(lambda r: int(industry_ct.loc[r["営業部"], r["業種大分類"]]), axis=1)
        industry_top["期待値"] = industry_top.apply(lambda r: float(expected_df.loc[r["営業部"], r["業種大分類"]]), axis=1)
        industry_top = industry_top.reindex(industry_top["残差"].abs().sort_values(ascending=False).index).head(10)

    industry_summary = (
        df.groupby(["営業部", "業種大分類"], as_index=False)
        .agg(件数=("売上高", "size"))
        .sort_values(["件数", "営業部"], ascending=[False, True])
    )

    # 売上高の検定
    revenue_groups = [g["売上高"].to_numpy(dtype=float) for _, g in df.groupby("営業部")]
    revenue_groups = [g for g in revenue_groups if len(g) >= STAT_TEST_MIN_CASES]
    revenue_detail = (
        df.groupby("営業部")["売上高"]
        .agg(["count", "mean", "median", "std"])
        .rename(columns={"count": "件数", "mean": "平均", "median": "中央値", "std": "標準偏差"})
        .reset_index()
    )
    revenue_summary = pd.DataFrame()
    if len(revenue_groups) >= 2:
        try:
            anova_f, anova_p = stats.f_oneway(*revenue_groups)
        except Exception:
            anova_f, anova_p = None, None
        try:
            kruskal_h, kruskal_p = stats.kruskal(*revenue_groups)
        except Exception:
            kruskal_h, kruskal_p = None, None
        all_vals = np.concatenate(revenue_groups)
        grand = float(np.mean(all_vals))
        ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in revenue_groups)
        ss_total = float(np.sum((all_vals - grand) ** 2))
        eta2 = ss_between / ss_total if ss_total > 0 else None
        eps2 = ((kruskal_h - len(revenue_groups) + 1) / (len(all_vals) - len(revenue_groups))) if (kruskal_h is not None and len(all_vals) > len(revenue_groups)) else None
        revenue_summary = pd.DataFrame(
            [
                {
                    "検定": "Kruskal-Wallis",
                    "統計量": float(kruskal_h) if kruskal_h is not None else None,
                    "p値": float(kruskal_p) if kruskal_p is not None else None,
                    "効果量": float(eps2) if eps2 is not None else None,
                    "補足": "売上高の群間差は分布が歪むためこちらを主に採用",
                },
                {
                    "検定": "ANOVA (log売上)",
                    "統計量": float(anova_f) if anova_f is not None else None,
                    "p値": float(anova_p) if anova_p is not None else None,
                    "効果量": float(eta2) if eta2 is not None else None,
                    "補足": "log1p売上で群間平均差を確認",
                },
            ]
        )
    else:
        revenue_summary = pd.DataFrame()

    sig_rows = []
    if industry_chi2 is not None:
        sig_rows.append(
            {
                "項目": "業種分布",
                "検定": "カイ二乗検定",
                "統計量": industry_chi2,
                "p値": industry_p,
                "効果量": industry_v,
                "補足": f"df={industry_dof}",
            }
        )
    if not revenue_summary.empty:
        for _, row in revenue_summary.iterrows():
            sig_rows.append(
                {
                    "項目": "売上高",
                    "検定": row["検定"],
                    "統計量": row["統計量"],
                    "p値": row["p値"],
                    "効果量": row["効果量"],
                    "補足": row["補足"],
                }
            )
    sig_summary = pd.DataFrame(sig_rows)

    stat_rows = []
    for case in all_cases:
        month = _case_month(case)
        if not month or month < MONTH_FILTER_START:
            continue
        status = case.get("final_status") or "未登録"
        if status not in ("成約", "失注"):
            continue
        dept = _normalize_sales_dept(case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or "")
        if not dept:
            continue
        row = {"営業部": dept, "月": month, "結果": status}
        for key, _ in STAT_TEST_METRICS:
            row[key] = _case_metric_value(case, key)
        stat_rows.append(row)
    stat_df = pd.DataFrame(stat_rows)

    return {
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "analysis_df": df,
        "industry_table": industry_summary,
        "industry_summary": pd.DataFrame(
            [
                {
                    "項目": "業種分布",
                    "検定": "カイ二乗検定",
                    "統計量": industry_chi2,
                    "p値": industry_p,
                    "効果量": industry_v,
                    "補足": f"df={industry_dof}",
                }
            ]
        )
        if industry_chi2 is not None
        else pd.DataFrame(),
        "industry_top": industry_top,
        "revenue_summary": revenue_summary,
        "revenue_detail": revenue_detail,
        "stat_df": stat_df,
        "sig_summary": sig_summary,
    }


def _build_gemini_prompt(
    dept_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    monthly_dept_summary: pd.DataFrame,
    significance_summary: pd.DataFrame,
) -> str:
    dept_text = dept_summary.head(12).to_string(index=False) if not dept_summary.empty else "データなし"
    industry_text = industry_summary.head(15).to_string(index=False) if not industry_summary.empty else "データなし"
    monthly_text = monthly_summary.head(18).to_string(index=False) if not monthly_summary.empty else "データなし"
    monthly_dept_text = monthly_dept_summary.head(20).to_string(index=False) if not monthly_dept_summary.empty else "データなし"
    sig_text = significance_summary.head(20).to_string(index=False) if not significance_summary.empty else "データなし"
    return f"""あなたはリース審査の実績分析担当です。以下の集計を読み、営業責任者向けに簡潔で実務的なコメントを日本語でまとめてください。

## 営業部別サマリ
{dept_text}

## 営業部×業種の上位分布
{industry_text}

## 月次推移
{monthly_text}

## 月次推移（営業部別の平均金利）
{monthly_dept_text}

## 営業部間の統計的有意差検定
{sig_text}

## 出力要件
1. 3〜6個の箇条書きで、重要な示唆を先に述べること
2. 営業部ごとの業種偏り、成約件数の強弱、月次の金利変動を必ず触れること
3. 営業部ごとの平均金利の変動差にも触れること
4. 各項目について、営業部間に有意差があるかを必ず触れること
5. 金利が上がっている月と下がっている月の要因仮説を1つずつ述べること
6. 最後に、次月に優先すべき営業アクションを1〜2行で書くこと
7. 断定しすぎず、データからの示唆として表現すること
"""


def _call_dashboard_gemini(prompt: str) -> str:
    api_key = (st.session_state.get("gemini_api_key", "").strip() or get_gemini_api_key() or "").strip()
    if not api_key:
        return "Gemini APIキーが設定されていません。サイドバーの AIモデル設定で入力してください。"
    model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT) or GEMINI_MODEL_DEFAULT
    if "1.5" in model:
        model = GEMINI_MODEL_DEFAULT
    resp = _gemini_chat(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=120,
        max_output_tokens=1400,
    )
    return (resp.get("message", {}) or {}).get("content", "") or "Gemini から空の応答が返されました。"


def _render_dept_analysis(df_dept: pd.DataFrame) -> None:
    st.subheader("🏢 営業部ごとの結果")
    if df_dept.empty:
        st.caption("営業部集計に使えるデータがありません。")
        return

    dept_summary = (
        df_dept
        .pivot_table(index="営業部", columns="結果", aggfunc="size", fill_value=0)
        .reset_index()
    )
    for col in ["成約", "失注", "保留", "未登録"]:
        if col not in dept_summary.columns:
            dept_summary[col] = 0
    dept_summary["合計"] = dept_summary[["成約", "失注", "保留", "未登録"]].sum(axis=1)
    dept_summary["成約率(%)"] = dept_summary.apply(
        lambda r: (r["成約"] / r["合計"] * 100) if r["合計"] else 0.0,
        axis=1,
    )
    dept_summary = dept_summary[["営業部", "成約", "失注", "保留", "未登録", "合計", "成約率(%)"]]
    existing_depts = set(dept_summary["営業部"].astype(str))
    missing_rows = [
        {"営業部": dept, "成約": 0, "失注": 0, "保留": 0, "未登録": 0, "合計": 0, "成約率(%)": 0.0}
        for dept in SALES_DEPT_OPTIONS
        if dept not in existing_depts
    ]
    if missing_rows:
        dept_summary = pd.concat([dept_summary, pd.DataFrame(missing_rows)], ignore_index=True)
    dept_summary = dept_summary.sort_values(by=["成約率(%)", "成約"], ascending=[False, False])

    st.dataframe(
        dept_summary.style.format({"成約率(%)": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### 📊 営業部ごとの差の検証（業種・成約分布）")
    dept_options = sorted(df_dept["営業部"].dropna().astype(str).unique().tolist())
    selected_dept = st.selectbox("比較する営業部", dept_options, key="dash_selected_dept")

    df_selected = df_dept[df_dept["営業部"] == selected_dept].copy()
    if df_selected.empty:
        st.caption("選択した営業部の案件データがありません。")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"{selected_dept}の業種分布")
        industry_dist = (
            df_selected.groupby("業種小分類").size().reset_index(name="件数").sort_values("件数", ascending=False).head(12)
        )
        st.bar_chart(industry_dist.set_index("業種小分類")["件数"], height=280)
        st.dataframe(industry_dist, width="stretch", hide_index=True)
    with c2:
        st.caption(f"{selected_dept}の結果分布")
        result_order = ["成約", "失注", "保留", "未登録"]
        result_dist = df_selected.groupby("結果").size().reindex(result_order, fill_value=0)
        st.bar_chart(result_dist, height=280)

    st.markdown("#### 📌 営業部×業種の成約傾向")
    dept_industry = (
        df_dept.groupby(["営業部", "業種大分類"], as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
    )
    dept_industry["成約率(%)"] = dept_industry.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )
    dept_industry = dept_industry.sort_values(["件数", "成約件数"], ascending=[False, False])
    st.dataframe(
        dept_industry.head(20).style.format({"成約率(%)": "{:.1f}", "平均金利": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )


def _render_monthly_analysis(df_monthly: pd.DataFrame) -> None:
    st.subheader("📅 月次推移")
    if df_monthly.empty:
        st.caption("2025年4月以降の成約・失注データがありません。")
        return

    monthly = (
        df_monthly.groupby("月", as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values("月")
    )
    monthly["成約率(%)"] = monthly.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )
    monthly["平均金利"] = monthly["平均金利"].fillna(0.0)

    if not monthly.empty:
        latest = monthly.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("直近月の件数", f"{int(latest['件数'])}件")
        c2.metric("直近月の成約件数", f"{int(latest['成約件数'])}件")
        c3.metric("直近月の平均金利", f"{latest['平均金利']:.2f}%", help="審査日基準・2025年4月以降の成約/失注を含む平均")

        monthly_dept = (
            df_monthly.groupby(["月", "営業部"], as_index=False)
            .agg(
                件数=("結果", "size"),
                成約件数=("成約フラグ", "sum"),
                平均金利=("金利", "mean"),
            )
            .sort_values(["月", "営業部"])
        )
        monthly_dept["成約率(%)"] = monthly_dept.apply(
            lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["月"],
            y=monthly["平均金利"],
            name="全体 平均金利(%)",
            mode="lines+markers",
            line=dict(color="#111827", width=4),
        ))
        dept_palette = {
            "宇都宮営業部": "#dc2626",
            "小山営業部": "#2563eb",
            "足利営業部": "#16a34a",
            "埼玉営業部": "#f59e0b",
            "未設定": "#7c3aed",
            "未読取": "#64748b",
        }
        for dept in [d for d in SALES_DEPT_OPTIONS if d in set(monthly_dept["営業部"].astype(str))]:
            dept_series = monthly_dept[monthly_dept["営業部"] == dept].set_index("月")["平均金利"]
            fig.add_trace(go.Scatter(
                x=dept_series.index,
                y=dept_series.values,
                name=f"{dept} 平均金利(%)",
                mode="lines+markers",
                line=dict(color=dept_palette.get(dept, "#64748b"), width=2),
            ))
        fig.update_layout(
            title="月次の平均金利（全体 + 営業部別）",
            xaxis_title="月",
            yaxis_title="平均金利(%)",
            legend=dict(orientation="h"),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📈 月次集計")
    st.dataframe(
        monthly.style.format({"平均金利": "{:.2f}", "成約率(%)": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### 🏢 営業部ごとの月次平均金利")
    dept_options = [d for d in SALES_DEPT_OPTIONS if d in set(df_monthly["営業部"].astype(str))] or sorted(df_monthly["営業部"].dropna().astype(str).unique().tolist())
    selected_dept = st.selectbox("営業部を選択", dept_options, key="dash_monthly_dept")

    dept_monthly = monthly_dept[monthly_dept["営業部"] == selected_dept].copy()
    if not dept_monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dept_monthly["月"], y=dept_monthly["件数"], name="件数", marker_color="#2563eb", opacity=0.35))
        fig.add_trace(go.Scatter(
            x=dept_monthly["月"],
            y=dept_monthly["平均金利"],
            name=f"{selected_dept} 平均金利(%)",
            mode="lines+markers",
            line=dict(color="#dc2626", width=3),
            yaxis="y2",
        ))
        fig.update_layout(
            title=f"{selected_dept} の月次平均金利（審査日基準 / 2025年4月以降）",
            xaxis_title="月",
            yaxis=dict(title="件数"),
            yaxis2=dict(title="平均金利(%)", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            dept_monthly[["月", "件数", "成約件数", "平均金利", "成約率(%)"]].style.format({"平均金利": "{:.2f}", "成約率(%)": "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("選択した営業部の月次データがありません。")

    st.markdown("#### 📊 営業部別の月次平均金利比較")
    if not monthly_dept.empty:
        compare = monthly_dept.pivot_table(index="月", columns="営業部", values="平均金利", aggfunc="mean").sort_index()
        compare = compare[[col for col in SALES_DEPT_OPTIONS if col in compare.columns] + [c for c in compare.columns if c not in SALES_DEPT_OPTIONS]]
        fig_cmp = go.Figure()
        for col in compare.columns:
            fig_cmp.add_trace(go.Scatter(x=compare.index, y=compare[col], mode="lines+markers", name=col))
        fig_cmp.update_layout(
            title="営業部別の平均金利推移",
            xaxis_title="月",
            yaxis_title="平均金利(%)",
            legend=dict(orientation="h"),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_cmp, width="stretch")


def _render_gemini_comment(df_dept: pd.DataFrame, df_monthly: pd.DataFrame, significance_summary: pd.DataFrame) -> None:
    st.subheader("🤖 Gemini コメント")
    if df_dept.empty or df_monthly.empty:
        st.caption("コメント生成に十分なデータがありません。")
        return

    dept_summary = (
        df_dept
        .groupby("営業部", as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
    )
    dept_summary["成約率(%)"] = dept_summary.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )
    dept_summary = dept_summary.sort_values(["成約件数", "件数"], ascending=[False, False])

    monthly_summary = (
        df_monthly.groupby("月", as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values("月")
    )
    monthly_summary["成約率(%)"] = monthly_summary.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )

    industry_summary = (
        df_dept.groupby(["営業部", "業種大分類"], as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values(["件数", "成約件数"], ascending=[False, False])
    )
    industry_summary["成約率(%)"] = industry_summary.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )

    monthly_dept_summary = (
        df_monthly.groupby(["月", "営業部"], as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values(["月", "営業部"])
    )
    monthly_dept_summary["成約率(%)"] = monthly_dept_summary.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )

    prompt = _build_gemini_prompt(dept_summary, industry_summary, monthly_summary, monthly_dept_summary, significance_summary)
    sig = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("✨ Gemini コメント生成", type="primary", key="dash_gemini_comment_run"):
            with st.spinner("Gemini にコメント生成を依頼中..."):
                comment = _call_dashboard_gemini(prompt)
            st.session_state["dashboard_gemini_comment"] = comment
            st.session_state["dashboard_gemini_comment_sig"] = sig
            st.rerun()
        if st.button("🗑️ コメントをクリア", key="dash_gemini_comment_clear"):
            st.session_state.pop("dashboard_gemini_comment", None)
            st.session_state.pop("dashboard_gemini_comment_sig", None)
            st.rerun()

    comment = st.session_state.get("dashboard_gemini_comment")
    comment_sig = st.session_state.get("dashboard_gemini_comment_sig")
    if comment:
        if comment_sig != sig:
            st.warning("集計データが変わっているため、コメントを再生成してください。")
        st.markdown(comment)
    else:
        st.caption("ボタンを押すと、営業部別・業種別・月次推移を踏まえた Gemini コメントを生成します。")


def _render_stat_significance_section(stat_result: dict | None) -> None:
    st.subheader("🔬 営業部間の統計的有意差")
    if not stat_result:
        st.caption("計算ボタンを押すと、業種分布と売上高の有意差を計算して表示します。")
        return

    st.caption(f"最終計算: {stat_result.get('computed_at', '不明')}  /  キャッシュ済み結果を表示中")

    industry_summary = stat_result.get("industry_summary", pd.DataFrame())
    revenue_summary = stat_result.get("revenue_summary", pd.DataFrame())
    industry_top = stat_result.get("industry_top", pd.DataFrame())
    revenue_detail = stat_result.get("revenue_detail", pd.DataFrame())
    sig_summary = stat_result.get("sig_summary", pd.DataFrame())

    if not sig_summary.empty:
        show = sig_summary.copy()
        for col in ["統計量", "p値", "効果量"]:
            if col in show.columns:
                show[col] = show[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.caption("表示できる検定結果がまだありません。")

    if not industry_summary.empty:
        st.markdown("#### 業種分布")
        row = industry_summary.iloc[0]
        st.write(
            f"カイ二乗検定: χ²={row['統計量']:.2f}, p={row['p値']:.4g}, Cramer's V={row['効果量']:.3f}, {row['補足']}"
        )
        st.dataframe(
            stat_result.get("industry_table", pd.DataFrame()).head(20),
            width="stretch",
            hide_index=True,
        )
        if not industry_top.empty:
            top_show = industry_top.copy()
            top_show["残差"] = top_show["残差"].map(lambda v: f"{v:.2f}")
            top_show["観測値"] = top_show["観測値"].map(lambda v: f"{int(v)}")
            top_show["期待値"] = top_show["期待値"].map(lambda v: f"{v:.1f}")
            st.caption("偏りが大きいセル（標準化残差の絶対値が大きい順）")
            st.dataframe(top_show, width="stretch", hide_index=True)

    if not revenue_summary.empty:
        st.markdown("#### 売上高")
        st.caption("売上高は分布が歪むため、Kruskal-Wallis を主、log売上のANOVA を補助で見ています。")
        show_rev = revenue_summary.copy()
        for col in ["統計量", "p値", "効果量"]:
            if col in show_rev.columns:
                show_rev[col] = show_rev[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        st.dataframe(show_rev, width="stretch", hide_index=True)
        if not revenue_detail.empty:
            st.dataframe(revenue_detail.style.format({"平均": "{:.3f}", "中央値": "{:.3f}", "標準偏差": "{:.3f}"}), width="stretch", hide_index=True)


def _render_stat_visualizations(stat_result: dict | None) -> None:
    if not stat_result:
        return

    analysis_df = stat_result.get("analysis_df", pd.DataFrame())
    stat_df = stat_result.get("stat_df", pd.DataFrame())
    if analysis_df.empty:
        return

    st.markdown("#### 図で見る営業部差")

    if not stat_result.get("industry_table", pd.DataFrame()).empty:
        industry_ct = pd.crosstab(analysis_df["営業部"], analysis_df["業種大分類"]).reindex(index=SALES_DEPT_OPTIONS, fill_value=0)
        pct = industry_ct.div(industry_ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
        c1, c2 = st.columns(2)
        with c1:
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pct.fillna(0).to_numpy(),
                    x=list(pct.columns),
                    y=list(pct.index),
                    colorscale="Blues",
                    colorbar=dict(title="構成比%"),
                    hovertemplate="営業部=%{y}<br>業種=%{x}<br>構成比=%{z:.1f}%<extra></extra>",
                )
            )
            fig_heat.update_layout(title="業種構成比ヒートマップ", height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_heat, width="stretch")
        with c2:
            fig_bar = go.Figure()
            for major in pct.columns:
                fig_bar.add_trace(
                    go.Bar(
                        x=list(pct.index),
                        y=pct[major].fillna(0),
                        name=major,
                    )
                )
            fig_bar.update_layout(
                barmode="stack",
                title="業種構成比（積み上げ）",
                xaxis_title="営業部",
                yaxis_title="構成比(%)",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_bar, width="stretch")

    if not stat_result.get("revenue_detail", pd.DataFrame()).empty:
        c1, c2 = st.columns(2)
        with c1:
            rev_df = analysis_df[["営業部", "売上高"]].dropna()
            fig_box = go.Figure()
            for dept in SALES_DEPT_OPTIONS:
                series = rev_df.loc[rev_df["営業部"] == dept, "売上高"]
                if series.empty:
                    continue
                fig_box.add_trace(go.Box(y=series, name=dept, boxmean=True))
            fig_box.update_layout(
                title="売上高の分布（営業部別）",
                yaxis_title="売上高",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_box, width="stretch")
        with c2:
            revenue_detail = stat_result["revenue_detail"].copy()
            fig_med = go.Figure()
            fig_med.add_trace(go.Bar(x=revenue_detail["営業部"], y=revenue_detail["中央値"], name="中央値"))
            fig_med.add_trace(go.Bar(x=revenue_detail["営業部"], y=revenue_detail["平均"], name="平均"))
            fig_med.update_layout(
                barmode="group",
                title="売上高の平均・中央値比較",
                xaxis_title="営業部",
                yaxis_title="売上高",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_med, width="stretch")

    if not stat_df.empty:
        st.markdown("#### 他の数値項目も同じ画面で比較")
        metric_label_map = {key: label for key, label in STAT_TEST_METRICS}
        available = [key for key, _ in STAT_TEST_METRICS if key in stat_df.columns and stat_df[key].notna().any()]
        if available:
            selected_metric = st.selectbox(
                "比較する項目",
                available,
                format_func=lambda k: metric_label_map.get(k, k),
                key="dash_stat_metric",
            )
            metric_df = stat_df[["営業部", selected_metric]].dropna()
            if not metric_df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    fig_metric = go.Figure()
                    for dept in SALES_DEPT_OPTIONS:
                        series = metric_df.loc[metric_df["営業部"] == dept, selected_metric]
                        if series.empty:
                            continue
                        fig_metric.add_trace(go.Box(y=series, name=dept, boxmean=True))
                    fig_metric.update_layout(
                        title=f"{metric_label_map.get(selected_metric, selected_metric)} の分布（営業部別）",
                        yaxis_title=metric_label_map.get(selected_metric, selected_metric),
                        height=380,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_metric, width="stretch")
                with c2:
                    metric_summary = (
                        metric_df.groupby("営業部")[selected_metric]
                        .agg(["count", "mean", "median"])
                        .rename(columns={"count": "件数", "mean": "平均", "median": "中央値"})
                        .reset_index()
                    )
                    st.dataframe(metric_summary.style.format({"平均": "{:.3f}", "中央値": "{:.3f}"}), width="stretch", hide_index=True)


def _build_dept_summary(df_dept: pd.DataFrame) -> pd.DataFrame:
    if df_dept.empty:
        return pd.DataFrame()
    dept_summary = (
        df_dept
        .pivot_table(index="営業部", columns="結果", aggfunc="size", fill_value=0)
        .reset_index()
    )
    for col in ["成約", "失注", "保留", "未登録"]:
        if col not in dept_summary.columns:
            dept_summary[col] = 0
    dept_summary["合計"] = dept_summary[["成約", "失注", "保留", "未登録"]].sum(axis=1)
    dept_summary["成約率(%)"] = dept_summary.apply(
        lambda r: (r["成約"] / r["合計"] * 100) if r["合計"] else 0.0,
        axis=1,
    )
    dept_summary = dept_summary[["営業部", "成約", "失注", "保留", "未登録", "合計", "成約率(%)"]]
    existing_depts = set(dept_summary["営業部"].astype(str))
    missing_rows = [
        {"営業部": dept, "成約": 0, "失注": 0, "保留": 0, "未登録": 0, "合計": 0, "成約率(%)": 0.0}
        for dept in SALES_DEPT_OPTIONS
        if dept not in existing_depts
    ]
    if missing_rows:
        dept_summary = pd.concat([dept_summary, pd.DataFrame(missing_rows)], ignore_index=True)
    dept_summary = dept_summary.sort_values(by=["成約率(%)", "成約"], ascending=[False, False])
    return dept_summary


def _build_monthly_summary(df_monthly: pd.DataFrame) -> pd.DataFrame:
    if df_monthly.empty:
        return pd.DataFrame()
    monthly = (
        df_monthly.groupby("月", as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values("月")
    )
    monthly["成約率(%)"] = monthly.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )
    monthly["平均金利"] = monthly["平均金利"].fillna(0.0)
    return monthly


def _build_monthly_dept_summary(df_monthly: pd.DataFrame) -> pd.DataFrame:
    if df_monthly.empty:
        return pd.DataFrame()
    monthly_dept = (
        df_monthly.groupby(["月", "営業部"], as_index=False)
        .agg(
            件数=("結果", "size"),
            成約件数=("成約フラグ", "sum"),
            平均金利=("金利", "mean"),
        )
        .sort_values(["月", "営業部"])
    )
    monthly_dept["成約率(%)"] = monthly_dept.apply(
        lambda r: (r["成約件数"] / r["件数"] * 100) if r["件数"] else 0.0,
        axis=1,
    )
    return monthly_dept


def render_dashboard():
    """📊 履歴分析・実績ダッシュボード タブのUIとロジックを描画する"""
    st.title("📊 履歴分析・実績ダッシュボード")
    analysis = run_contract_driver_analysis()
    
    if analysis is None:
        st.warning("成約データが5件以上貯まると表示されます。結果登録で「成約」を登録してください。")
    else:
        n = analysis["closed_count"]
        st.success(f"成約 {n} 件を分析しました。")
        try:
            pdf_bytes = build_contract_report_pdf(analysis)
            filename = f"成約の正体レポート_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("📥 分析結果をPDFでダウンロード", data=pdf_bytes, file_name=filename, mime="application/pdf", key="dl_contract_report_pdf")
        except Exception as e:
            st.caption(f"PDF生成をスキップしました: {e}")
        st.divider()
        
        # ---------- 成約要因分析 ----------
        st.subheader("📈 成約要因分析")
        st.caption("成約した案件だけを抽出し、共通項と成約に効く因子を分析した結果です。")
        st.markdown("**成約に最も寄与している上位3つの因子（ドライバー）**")
        for i, d in enumerate(analysis["top3_drivers"], 1):
            st.markdown(f"**{i}. {d['label']}** … 係数 {d['coef']:.4f}（{d['direction']}に効く）")
        st.divider()
        
        st.subheader("成約案件の平均的な財務数値")
        if analysis["avg_financials"]:
            rows = []
            for k, v in analysis["avg_financials"].items():
                if "自己資本" in k:
                    rows.append({"指標": k, "平均値": f"{v:.1f}%"})
                elif isinstance(v, float) and abs(v) >= 1:
                    rows.append({"指標": k, "平均値": f"{v:,.0f}"})
                else:
                    rows.append({"指標": k, "平均値": f"{v:.4f}"})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.caption("財務データが取得できませんでした。")
        st.divider()
        
        st.subheader("成約案件で頻出する定性タグ（ランキング）")
        if analysis["tag_ranking"]:
            for rank, (tag, count) in enumerate(analysis["tag_ranking"], 1):
                st.markdown(f"{rank}. **{tag}** … {count}件")
        else:
            st.caption("定性タグの登録がありません。")
            
        # 定性スコアリングの集計（成約案件）
        st.divider()
        st.subheader("成約案件の定性スコアリング")
        qs = analysis.get("qualitative_summary")
        if qs and (qs.get("avg_weighted") is not None or qs.get("avg_combined") is not None or qs.get("rank_distribution")):
            n_qual = qs.get("n_with_qual", 0)
            st.caption(f"成約{n}件のうち、定性スコアリングを入力していた案件: **{n_qual}件**")
            if qs.get("avg_weighted") is not None:
                st.metric("定性スコア（加重）の平均", f"{qs['avg_weighted']:.1f} / 100", help="項目別5段階の加重平均")
            if qs.get("avg_combined") is not None:
                st.metric("合計（総合×重み＋定性×重み）の平均", f"{qs['avg_combined']:.1f}", help="ランク算出の元となる合計点")
            if qs.get("rank_distribution"):
                st.markdown("**ランク（A〜E）の分布**")
                for r, cnt in sorted(qs["rank_distribution"].items(), key=lambda x: (-x[1], x[0])):
                    st.caption(f"- **{r}** … {cnt}件")
        else:
            st.caption("定性スコアリングを入力した成約案件がまだありません。審査入力で「定性スコアリング」を選択し、結果登録で成約にするとここに集計が表示されます。")


    all_cases = load_all_cases()

    if all_cases:
        stat_result = st.session_state.get("dashboard_dept_stat_result")
        stat_key = hashlib.sha256(
            "|".join(
                f"{case.get('id','')}:{_case_timestamp(case)}:{case.get('final_status','')}"
                f":{case.get('sales_dept') or case.get('inputs', {}).get('sales_dept') or ''}"
                f":{case.get('industry_major') or case.get('inputs', {}).get('industry_major') or ''}"
                f":{case.get('industry_sub') or case.get('inputs', {}).get('industry_sub') or ''}"
                f":{case.get('final_rate') or case.get('result', {}).get('final_rate') or case.get('inputs', {}).get('final_rate') or ''}"
                f":{case.get('inputs', {}).get('nenshu') or case.get('financials', {}).get('nenshu') or ''}"
                f":{case.get('inputs', {}).get('gross_profit') or case.get('financials', {}).get('gross_profit') or ''}"
                f":{case.get('inputs', {}).get('op_profit') or case.get('financials', {}).get('op_profit') or ''}"
            for case in all_cases
        ).encode("utf-8")
        ).hexdigest()
        cached_key = st.session_state.get("dashboard_dept_stat_result_key")
        if stat_result is not None and cached_key != stat_key:
            stat_result = None
            st.session_state.pop("dashboard_dept_stat_result", None)
            st.session_state.pop("dashboard_dept_stat_result_key", None)

        control_left, control_right = st.columns([1, 3])
        with control_left:
            if st.button("🧮 業種分布と売上高を計算", type="primary", key="dash_dept_stat_run"):
                with st.spinner("重い集計を計算中..."):
                    stat_result = _compute_department_stat_analysis(all_cases)
                st.session_state["dashboard_dept_stat_result"] = stat_result
                st.session_state["dashboard_dept_stat_result_key"] = stat_key
                st.rerun()
            if st.button("🗑️ 計算結果をクリア", key="dash_dept_stat_clear"):
                st.session_state.pop("dashboard_dept_stat_result", None)
                st.session_state.pop("dashboard_dept_stat_result_key", None)
                st.rerun()
        with control_right:
            if stat_result is None:
                st.caption("計算ボタンを押すと、業種分布と売上高の営業部差を検定して結果をキャッシュします。")
            else:
                st.caption(f"キャッシュ済み結果を表示中 / 計算キー: {stat_key[:12]}")

        df_dept = _build_dept_industry_frame(all_cases)
        df_monthly = _build_monthly_frame(all_cases)
        dept_summary = _build_dept_summary(df_dept)
        monthly_summary = _build_monthly_summary(df_monthly)
        monthly_dept_summary = _build_monthly_dept_summary(df_monthly)
        st.divider()
        _render_stat_significance_section(stat_result)
        _render_stat_visualizations(stat_result)
        tabs = st.tabs(["🏢 営業部別", "📅 月次推移", "🤖 Gemini コメント"])
        with tabs[0]:
            _render_dept_analysis(df_dept)
        with tabs[1]:
            _render_monthly_analysis(df_monthly)
        with tabs[2]:
            _render_gemini_comment(df_dept, df_monthly, stat_result.get("sig_summary", pd.DataFrame()) if stat_result else pd.DataFrame())
    else:
        st.caption("まだ案件履歴がありません。")

    # ---------------- 案件履歴一覧 ----------------
    st.divider()
    st.subheader("📋 最新の案件履歴")
    if all_cases:
        for case in reversed(all_cases[-15:]):  # 最新15件を表示
            c_date = case.get('timestamp', '')[:16]
            c_sub = case.get('industry_sub', '不明')
            c_score = case.get('result', {}).get('score', 0)
            c_status = case.get('final_status', '未登録')
            title_emoji = "✅" if c_status == "成約" else "❌" if c_status == "失注" else "📝"
            with st.expander(f"{title_emoji} {c_date} - {c_sub} (スコア: {c_score:.0f}) [{c_status}]"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write(f"**判定**: {case.get('result', {}).get('hantei', '不明')}")
                    if case.get('chat_summary'):
                        st.caption(case['chat_summary'])
                    if st.button("🔄 このデータを入力に復元", key=f"restore_hist_{case.get('id', c_date)}"):
                        i_data = case.get('inputs', {})
                        st.session_state["last_submitted_inputs"] = {
                            "selected_major": case.get("industry_major", ""),
                            "selected_sub": case.get("industry_sub", ""),
                        }
                        for k, v in i_data.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    st.session_state[sub_k] = sub_v
                            else:
                                st.session_state[k] = v
                        st.session_state["main_mode"] = "📋 審査・分析"
                        st.session_state["nav_index"] = 0
                        st.rerun()
                with c2:
                    if case.get('ai_industry_advice'):
                        st.markdown("##### 📈 AI業界分析アドバイス")
                        st.info(case['ai_industry_advice'])
                    if case.get('ai_byoki'):
                        st.markdown("##### 🤖 AIのぼやき")
                        st.caption(case['ai_byoki'])
    else:
        st.caption("まだ案件履歴がありません。")
