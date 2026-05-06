"""
モデル見直しフックの実行基盤。

用途:
- 直近AUCの悪化検知
- 既存先 / 新規先の乖離検知
- 新規先専用特徴の効果確認

hooks/hooks.json に定義されたフックを実行し、結果を data/model_review_runs.jsonl に記録する。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HOOKS_FILE = os.path.join(_SCRIPT_DIR, "hooks", "hooks.json")
_RUNS_FILE = os.path.join(_SCRIPT_DIR, "data", "model_review_runs.jsonl")
_STATE_FILE = os.path.join(_SCRIPT_DIR, "data", "model_review_state.json")


def _load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_hook_definitions() -> list[dict]:
    data = _load_json(_HOOKS_FILE, {"hooks": []})
    hooks = data.get("hooks") or []
    return [h for h in hooks if isinstance(h, dict)]


def _append_run_log(record: dict) -> None:
    os.makedirs(os.path.dirname(_RUNS_FILE), exist_ok=True)
    with open(_RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_auc(y_true, y_score) -> float | None:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 2 or len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _load_closed_cases() -> list[dict]:
    from data_cases import load_all_cases

    cases = [c for c in load_all_cases() if c.get("final_status") in ("成約", "失注")]
    cases.sort(key=lambda c: str(c.get("timestamp") or c.get("final_result_date") or ""))
    return cases


def _case_score(case: dict, target: str = "score_borrower") -> float | None:
    result = case.get("result") or {}
    if target == "score_borrower":
        for key in ("score_borrower", "score"):
            val = result.get(key) if result.get(key) is not None else case.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None
    val = result.get(target) if result.get(target) is not None else case.get(target)
    return float(val) if isinstance(val, (int, float)) else None


def _industry_base_from_case(case: dict) -> str:
    """案件から業種ベースを返す。"""
    result = case.get("result") or {}
    major = result.get("industry_major") or case.get("industry_major") or "D 建設業"
    major_code = major.split(" ")[0] if isinstance(major, str) and " " in major else (major[0] if major else "D")
    if major_code == "H":
        return "運送業"
    if major_code == "P":
        return "医療"
    if major_code in ["I", "K", "M", "R"]:
        return "サービス業"
    if major_code == "E":
        return "製造業"
    return "全体"


def _normalize_sales_dept(value: object) -> str:
    dept = str(value or "").strip()
    if dept in ("", "0", "未設定", "未読取"):
        return ""
    valid = {"宇都宮営業部", "小山営業部", "足利営業部", "埼玉営業部"}
    return dept if dept in valid else ""


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


def _case_major(case: dict) -> str:
    result = case.get("result") or {}
    major = result.get("industry_major") or case.get("industry_major") or case.get("inputs", {}).get("industry_major") or ""
    return str(major or "不明")


def _case_revenue(case: dict) -> float | None:
    result = case.get("result") or {}
    inputs = case.get("inputs") or {}
    for key in ("nenshu", "sales", "revenue", "売上高"):
        val = result.get(key)
        if val is None:
            val = case.get(key)
        if val is None:
            val = inputs.get(key)
        num = _safe_float(val)
        if num is not None:
            return num
    return None


def _case_final_rate(case: dict) -> float | None:
    result = case.get("result") or {}
    inputs = case.get("inputs") or {}
    for val in (case.get("final_rate"), result.get("final_rate"), inputs.get("final_rate")):
        num = _safe_float(val)
        if num is not None:
            return num
    return None


def _case_score_value(case: dict, target: str = "score_borrower") -> float | None:
    result = case.get("result") or {}
    if target == "score_borrower":
        for key in ("score_borrower", "score"):
            val = result.get(key)
            if val is None:
                val = case.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None
    val = result.get(target)
    if val is None:
        val = case.get(target)
    return float(val) if isinstance(val, (int, float)) else None


def _kruskal_or_anova(groups: list[np.ndarray]) -> dict:
    from scipy import stats

    clean = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return {}

    try:
        h, p = stats.kruskal(*clean)
    except Exception:
        h, p = None, None
    try:
        f, p_a = stats.f_oneway(*clean)
    except Exception:
        f, p_a = None, None

    all_vals = np.concatenate(clean)
    n = len(all_vals)
    k = len(clean)
    eps2 = None
    if h is not None and n > k:
        eps2 = (float(h) - k + 1) / (n - k)

    eta2 = None
    if f is not None:
        grand = float(np.mean(all_vals))
        ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in clean)
        ss_total = float(np.sum((all_vals - grand) ** 2))
        if ss_total > 0:
            eta2 = ss_between / ss_total

    return {
        "kruskal_h": float(h) if h is not None else None,
        "kruskal_p": float(p) if p is not None else None,
        "anova_f": float(f) if f is not None else None,
        "anova_p": float(p_a) if p_a is not None else None,
        "epsilon2": float(eps2) if eps2 is not None else None,
        "eta2": float(eta2) if eta2 is not None else None,
        "n_groups": len(clean),
        "n_total": int(n),
    }


def _evaluate_department_significance(hook: dict) -> dict:
    """営業部ごとの統計的有意差をまとめて確認する。"""
    import pandas as pd
    from scipy import stats

    cases = _load_closed_cases()
    min_cases = int(hook.get("min_cases_per_dept", 8))
    alpha = float(hook.get("alpha", 0.05))
    min_effect = float(hook.get("min_effect", 0.05))

    rows = []
    for case in cases:
        dept = _normalize_sales_dept(case.get("sales_dept") or (case.get("inputs") or {}).get("sales_dept") or "")
        if not dept:
            continue
        rows.append(
            {
                "営業部": dept,
                "業種大分類": _case_major(case),
                "売上高": _case_revenue(case),
                "金利": _case_final_rate(case),
                "score": _case_score_value(case, hook.get("target", "score_borrower")),
            }
        )

    if not rows:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "営業部データが不足",
        }

    df = pd.DataFrame(rows)
    results = []
    dept_summary = (
        df.groupby("営業部", as_index=False)
        .agg(
            件数=("営業部", "size"),
            スコア平均=("score", "mean"),
            金利平均=("金利", "mean"),
            売上高平均=("売上高", "mean"),
            業種数=("業種大分類", pd.Series.nunique),
        )
        .sort_values(["件数", "営業部"], ascending=[False, True])
    )

    industry_ct = pd.crosstab(df["営業部"], df["業種大分類"])
    industry_rows = []
    if industry_ct.shape[0] >= 2 and industry_ct.shape[1] >= 2:
        chi2, p_value, dof, expected = stats.chi2_contingency(industry_ct)
        total_n = float(industry_ct.to_numpy().sum())
        cramers_v = float(np.sqrt(chi2 / (total_n * (min(industry_ct.shape) - 1)))) if total_n > 0 else None
        expected_df = pd.DataFrame(expected, index=industry_ct.index, columns=industry_ct.columns)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = (industry_ct - expected_df) / np.sqrt(expected_df)
        industry_rows = (
            residual.stack()
            .rename("残差")
            .reset_index()
            .rename(columns={"level_0": "営業部", "level_1": "業種大分類"})
        ).to_dict(orient="records")
        results.append({
            "項目": "業種分布",
            "検定": "カイ二乗検定",
            "統計量": float(chi2),
            "p値": float(p_value),
            "効果量": cramers_v,
            "補足": f"df={int(dof)} / 営業部間で業種構成が異なるか",
        })

    for col, label in [("score", "スコア"), ("金利", "金利"), ("売上高", "売上高")]:
        sub = df[["営業部", col]].dropna()
        if sub.empty:
            continue
        groups = [g[col].to_numpy(dtype=float) for _, g in sub.groupby("営業部")]
        groups = [g for g in groups if len(g) >= min_cases]
        if len(groups) < 2:
            continue
        test = _kruskal_or_anova(groups)
        if not test:
            continue
        detail = sub.groupby("営業部")[col].agg(["count", "mean", "median"]).rename(columns={"count": "件数", "mean": "平均", "median": "中央値"}).reset_index()
        top = detail.sort_values("平均", ascending=False).iloc[0]
        bottom = detail.sort_values("平均", ascending=True).iloc[0]
        results.append({
            "項目": label,
            "検定": "Kruskal-Wallis",
            "統計量": test.get("kruskal_h"),
            "p値": test.get("kruskal_p"),
            "効果量": test.get("epsilon2"),
            "補足": f"最上位={top['営業部']} / 最下位={bottom['営業部']}",
        })
        results.append({
            "項目": f"{label} (ANOVA)",
            "検定": "ANOVA",
            "統計量": test.get("anova_f"),
            "p値": test.get("anova_p"),
            "効果量": test.get("eta2"),
            "補足": f"最上位={top['営業部']} / 最下位={bottom['営業部']}",
        })

    if not results:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "有意差検定に必要なデータが不足",
        }

    sig_rows = []
    triggered = False
    for r in results:
        p = r.get("p値")
        eff = r.get("効果量")
        sig = (p is not None and p < alpha) and (eff is not None and eff >= min_effect)
        triggered = triggered or sig
        sig_rows.append({**r, "有意": "有意" if sig else "非有意"})

    message_parts = []
    for r in sig_rows:
        p = r.get("p値")
        eff = r.get("効果量")
        p_txt = f"{p:.4g}" if isinstance(p, (int, float)) else "—"
        eff_txt = f"{eff:.3f}" if isinstance(eff, (int, float)) else "—"
        message_parts.append(f"{r['項目']} p={p_txt} / 効果量={eff_txt}")

    return {
        "hook_id": hook.get("id"),
        "kind": hook.get("kind"),
        "status": "triggered" if triggered else "passed",
        "passed": not triggered,
        "results": sig_rows,
        "dept_summary": dept_summary.to_dict(orient="records"),
        "industry_table": industry_ct.reset_index().to_dict(orient="records"),
        "industry_residuals": industry_rows,
        "thresholds": {
            "alpha": alpha,
            "min_effect": min_effect,
            "min_cases_per_dept": min_cases,
        },
        "message": " | ".join(message_parts),
        "explanation": [
            "営業部ごとの業種構成・スコア・金利・売上をまとめて検定しています。",
            "p値が低く、効果量も十分な項目は営業部差があるため、補正や分割の見直し候補です。",
        ],
        "recommendation": "営業部別の閾値・係数・運用分岐を見直す" if triggered else "現状では営業部差は小さい",
    }


def _evaluate_recent_auc_drop(hook: dict) -> dict:
    cases = _load_closed_cases()
    min_cases = int(hook.get("min_cases", 40))
    if len(cases) < min_cases:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": f"件数不足 ({len(cases)}件 / 最低 {min_cases}件)",
        }

    scores = []
    labels = []
    for c in cases:
        s = _case_score(c, hook.get("target", "score_borrower"))
        if s is None:
            continue
        scores.append(s)
        labels.append(1 if c.get("final_status") == "成約" else 0)

    overall_auc = _safe_auc(labels, scores)
    if overall_auc is None:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "AUC算出に必要な両クラスが不足",
        }

    window = int(hook.get("recent_window", 60))
    recent_cases = cases[-window:]
    recent_scores, recent_labels = [], []
    for c in recent_cases:
        s = _case_score(c, hook.get("target", "score_borrower"))
        if s is None:
            continue
        recent_scores.append(s)
        recent_labels.append(1 if c.get("final_status") == "成約" else 0)
    recent_auc = _safe_auc(recent_labels, recent_scores)
    if recent_auc is None:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "直近AUC算出に必要な両クラスが不足",
            "details": {"overall_auc": overall_auc},
        }

    max_drop = float(hook.get("max_drop", 0.02))
    min_auc = float(hook.get("min_auc", 0.64))
    drop = overall_auc - recent_auc
    passed = not (recent_auc < min_auc or drop > max_drop)

    return {
        "hook_id": hook.get("id"),
        "kind": hook.get("kind"),
        "status": "passed" if passed else "triggered",
        "passed": passed,
        "overall_auc": round(overall_auc, 4),
        "recent_auc": round(recent_auc, 4),
        "drop": round(drop, 4),
        "thresholds": {"min_auc": min_auc, "max_drop": max_drop, "window": window},
        "message": (
            f"全体AUC {overall_auc:.3f} / 直近AUC {recent_auc:.3f} / 乖離 {drop:+.3f}"
        ),
        "explanation": [
            f"直近 {window} 件のAUCが全体AUCから {drop:+.3f} ずれています。",
            "直近AUCが閾値未満、または乖離が大きい場合は再学習候補です。",
        ],
        "recommendation": "直近劣化が続く場合は再学習候補" if not passed else "現状維持",
    }


def _evaluate_segment_gap(hook: dict) -> dict:
    cases = _load_closed_cases()
    field = hook.get("segment_field", "customer_type")
    segments = hook.get("segments") or ["既存先", "新規先"]
    min_cases = int(hook.get("min_cases_per_segment", 40))
    target = hook.get("target", "score_borrower")
    rows = []
    for seg in segments:
        seg_cases = [c for c in cases if (c.get(field) or (c.get("inputs") or {}).get(field) or "既存先") == seg]
        if len(seg_cases) < min_cases:
            rows.append({"segment": seg, "status": "skipped", "n": len(seg_cases)})
            continue
        labels = []
        scores = []
        for c in seg_cases:
            s = _case_score(c, target)
            if s is None:
                continue
            labels.append(1 if c.get("final_status") == "成約" else 0)
            scores.append(s)
        auc = _safe_auc(labels, scores)
        rows.append({"segment": seg, "status": "ok" if auc is not None else "skipped", "n": len(labels), "auc": auc})

    aucs = [r["auc"] for r in rows if r.get("auc") is not None]
    if len(aucs) < 2:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "比較できるセグメントが不足",
            "details": rows,
        }
    gap = abs(aucs[0] - aucs[1])
    max_gap = float(hook.get("max_gap", 0.05))
    passed = gap <= max_gap
    return {
        "hook_id": hook.get("id"),
        "kind": hook.get("kind"),
        "status": "passed" if passed else "triggered",
        "passed": passed,
        "gap": round(gap, 4),
        "thresholds": {"max_gap": max_gap},
        "segments": rows,
        "message": " / ".join(
            f"{r['segment']}={r.get('auc', None):.3f}({r.get('n', 0)}件)" if r.get("auc") is not None else f"{r['segment']}=—({r.get('n', 0)}件)"
            for r in rows
        ),
        "explanation": [
            f"指定セグメント間のAUC差は {gap:.3f} です。",
            "既存先 / 新規先のどちらかが大きく弱い場合、分割モデルや特徴の見直し候補です。",
        ],
        "recommendation": "セグメント別モデル検討" if not passed else "現状維持",
    }


def _evaluate_industry_monitor(hook: dict) -> dict:
    """業種別AUCとベンチ/業種乖離を同じフレームで評価する。"""
    cases = _load_closed_cases()
    industry_bases = hook.get("industry_bases") or ["全体", "医療", "運送業", "サービス業", "製造業"]
    min_cases = int(hook.get("min_cases_per_industry", 40))
    min_auc = float(hook.get("min_auc", 0.64))
    max_mean_gap = float(hook.get("max_mean_gap_pt", 12.0))
    max_p90_gap = float(hook.get("max_p90_gap_pt", 20.0))

    rows: list[dict] = []
    for base in industry_bases:
        seg_cases = cases if base == "全体" else [c for c in cases if _industry_base_from_case(c) == base]
        labels = []
        scores = []
        gap_vals = []
        signed_gap_vals = []
        for c in seg_cases:
            s = _case_score(c, hook.get("target", "score_borrower"))
            if s is not None:
                scores.append(s)
                labels.append(1 if c.get("final_status") == "成約" else 0)
            res = c.get("result") or {}
            bench = res.get("bench_score")
            ind = res.get("ind_score")
            if isinstance(bench, (int, float)) and isinstance(ind, (int, float)):
                gap = float(bench) - float(ind)
                gap_vals.append(abs(gap))
                signed_gap_vals.append(gap)

        auc = _safe_auc(labels, scores) if len(labels) >= min_cases else None
        row = {
            "industry": base,
            "n": len(labels),
            "auc": auc,
            "bench_ind_gap_mean": float(np.mean(gap_vals)) if gap_vals else None,
            "bench_ind_gap_median": float(np.median(gap_vals)) if gap_vals else None,
            "bench_ind_gap_p90": float(np.percentile(gap_vals, 90)) if gap_vals else None,
            "bench_ind_gap_signed_mean": float(np.mean(signed_gap_vals)) if signed_gap_vals else None,
        }
        rows.append(row)

    valid_auc_rows = [r for r in rows if r.get("auc") is not None]
    if not valid_auc_rows:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "業種別AUC算出に必要な件数が不足",
            "details": rows,
        }

    worst_auc_row = min(valid_auc_rows, key=lambda r: r["auc"])
    gap_rows = [r for r in rows if r.get("bench_ind_gap_mean") is not None]
    worst_gap_row = max(gap_rows, key=lambda r: r["bench_ind_gap_mean"], default=None)
    worst_p90_row = max(gap_rows, key=lambda r: r["bench_ind_gap_p90"], default=None)

    passed = True
    if worst_auc_row["auc"] is not None and worst_auc_row["auc"] < min_auc:
        passed = False
    if worst_gap_row and worst_gap_row["bench_ind_gap_mean"] is not None and worst_gap_row["bench_ind_gap_mean"] > max_mean_gap:
        passed = False
    if worst_p90_row and worst_p90_row["bench_ind_gap_p90"] is not None and worst_p90_row["bench_ind_gap_p90"] > max_p90_gap:
        passed = False

    msg_parts = []
    for row in rows:
        auc_part = f"{row['auc']:.3f}" if row.get("auc") is not None else "—"
        gap_part = f"{row['bench_ind_gap_mean']:.1f}pt" if row.get("bench_ind_gap_mean") is not None else "—"
        msg_parts.append(f"{row['industry']} AUC {auc_part} / bench-ind {gap_part}")

    return {
        "hook_id": hook.get("id"),
        "kind": hook.get("kind"),
        "status": "passed" if passed else "triggered",
        "passed": passed,
        "rows": rows,
        "thresholds": {
            "min_auc": min_auc,
            "max_mean_gap_pt": max_mean_gap,
            "max_p90_gap_pt": max_p90_gap,
            "min_cases_per_industry": min_cases,
        },
        "message": " | ".join(msg_parts),
        "explanation": [
            "業種ごとのAUCと bench/ind 乖離を同じ一覧で確認しています。",
            "AUCが低い業種、または bench/ind 乖離が大きい業種は補正候補です。",
        ],
        "recommendation": "業種別の再学習・乖離補正を検討" if not passed else "現状維持",
    }


def _evaluate_feature_ab_test(hook: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from data_cases import load_all_cases
    from analysis_regression import build_design_matrix_from_logs, COEFF_MAIN_KEYS, COEFF_EXTRA_KEYS

    model_key = hook.get("segment_model_key", "全体_新規先")
    X, y = build_design_matrix_from_logs(load_all_cases(), model_key=model_key)
    if X is None or y is None or len(y) < int(hook.get("min_cases", 40)):
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "学習件数不足",
        }

    feature_names = COEFF_MAIN_KEYS + COEFF_EXTRA_KEYS
    baseline_prefixes = tuple(hook.get("baseline_prefixes") or ["new_customer_"])
    baseline_idx = [i for i, name in enumerate(feature_names) if not name.startswith(baseline_prefixes)]
    if not baseline_idx:
        baseline_idx = list(range(X.shape[1]))

    def _cv_auc(X_use):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        class_counts = np.bincount(y, minlength=2)
        max_splits = int(min(int(hook.get("cv_folds", 5)), int(class_counts.min())))
        if max_splits < 2:
            return None, None
        cv = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in cv.split(X_use, y):
            model = LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_use[tr], y[tr])
            prob = model.predict_proba(X_use[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        return float(np.mean(aucs)), float(np.std(aucs))

    base_auc, base_std = _cv_auc(X[:, baseline_idx])
    cand_auc, cand_std = _cv_auc(X)
    if base_auc is None or cand_auc is None:
        return {
            "hook_id": hook.get("id"),
            "kind": hook.get("kind"),
            "status": "skipped",
            "message": "CV分割に必要なクラス数が不足",
        }
    delta = cand_auc - base_auc
    min_gain = float(hook.get("min_gain", 0.01))
    passed = delta >= min_gain
    return {
        "hook_id": hook.get("id"),
        "kind": hook.get("kind"),
        "status": "passed" if passed else "triggered",
        "passed": passed,
        "baseline_auc": round(base_auc, 4),
        "candidate_auc": round(cand_auc, 4),
        "delta": round(delta, 4),
        "thresholds": {"min_gain": min_gain, "cv_folds": int(hook.get("cv_folds", 5))},
        "message": f"baseline {base_auc:.3f} → candidate {cand_auc:.3f} (Δ={delta:+.3f})",
        "recommendation": "新規先特徴を採用" if passed else "特徴追加を再検討",
    }


def evaluate_hook(hook: dict) -> dict:
    kind = hook.get("kind")
    if kind == "recent_auc_drop":
        return _evaluate_recent_auc_drop(hook)
    if kind == "segment_auc_gap":
        return _evaluate_segment_gap(hook)
    if kind == "industry_monitor":
        return _evaluate_industry_monitor(hook)
    if kind == "department_significance":
        return _evaluate_department_significance(hook)
    if kind == "feature_ab_test":
        return _evaluate_feature_ab_test(hook)
    return {
        "hook_id": hook.get("id"),
        "kind": kind,
        "status": "skipped",
        "message": f"未対応のkind: {kind}",
        "explanation": ["このフック種別はまだ実装されていません。"],
    }


def run_model_review_hooks(force: bool = False) -> dict:
    """設定済みフックをまとめて実行し、結果を保存して返す。"""
    hooks = [h for h in load_hook_definitions() if h.get("enabled", True)]
    results = [evaluate_hook(h) for h in hooks]
    summary = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "force": bool(force),
        "total": len(results),
        "triggered": sum(1 for r in results if r.get("status") == "triggered"),
        "passed": sum(1 for r in results if r.get("status") == "passed"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "results": results,
    }
    _append_run_log(summary)
    _save_json(_STATE_FILE, summary)
    return summary


def get_model_review_hook_status() -> dict:
    hooks = load_hook_definitions()
    state = _load_json(_STATE_FILE, {})
    return {
        "hooks": hooks,
        "state": state,
        "hook_count": len(hooks),
    }


def render_model_review_hook_panel() -> None:
    """Streamlit 用の軽い操作パネル。"""
    try:
        import streamlit as st
    except Exception:
        return

    status = get_model_review_hook_status()
    state = status.get("state") or {}
    with st.container(border=True):
        st.markdown("#### 🪝 モデル見直しフック")
        c1, c2, c3 = st.columns(3)
        c1.metric("フック数", f"{status.get('hook_count', 0)}")
        c2.metric("直近実行", state.get("ts", "未実行"))
        c3.metric("トリガー", f"{state.get('triggered', 0)}件")

        if st.button("▶ フックを実行", key="btn_run_model_review_hooks", type="primary"):
            with st.spinner("モデル見直しフックを実行中..."):
                res = run_model_review_hooks(force=True)
            st.success(
                f"実行完了: {res['passed']}件通過 / {res['triggered']}件トリガー / {res['skipped']}件スキップ"
            )
            with st.expander("フック結果", expanded=True):
                for item in res.get("results", []):
                    st.markdown(f"**{item.get('hook_id')}**: {item.get('status')} — {item.get('message')}")
                    if item.get("explanation"):
                        with st.expander("説明", expanded=False):
                            for line in item.get("explanation", []):
                                st.markdown(f"- {line}")
                    if item.get("kind") == "department_significance" and item.get("results"):
                        import pandas as pd
                        import plotly.graph_objects as go
                        df = pd.DataFrame(item["results"])
                        if not df.empty:
                            show = df.copy()
                            for col in ["統計量", "p値", "効果量"]:
                                if col in show.columns:
                                    show[col] = show[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
                            st.dataframe(show, width='stretch', hide_index=True)
                        dept_summary = pd.DataFrame(item.get("dept_summary") or [])
                        if not dept_summary.empty:
                            st.markdown("##### 営業部別の平均値")
                            dept_summary = dept_summary.sort_values("件数", ascending=False)
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name="スコア平均", x=dept_summary["営業部"], y=dept_summary["スコア平均"], marker_color="#2563eb"))
                            fig.add_trace(go.Bar(name="金利平均", x=dept_summary["営業部"], y=dept_summary["金利平均"], marker_color="#f97316"))
                            fig.update_layout(
                                barmode="group",
                                height=360,
                                margin=dict(l=20, r=20, t=30, b=20),
                                legend=dict(orientation="h"),
                                yaxis_title="平均値",
                            )
                            st.plotly_chart(fig, width='stretch', use_container_width=True)

                            fig_rev = go.Figure()
                            fig_rev.add_trace(go.Bar(
                                name="売上高平均",
                                x=dept_summary["営業部"],
                                y=dept_summary["売上高平均"],
                                marker_color="#16a34a",
                            ))
                            fig_rev.update_layout(
                                height=320,
                                margin=dict(l=20, r=20, t=30, b=20),
                                yaxis_title="売上高平均",
                            )
                            st.plotly_chart(fig_rev, width='stretch', use_container_width=True)

                        ind_rows = pd.DataFrame(item.get("industry_table") or [])
                        if not ind_rows.empty and "営業部" in ind_rows.columns:
                            st.markdown("##### 営業部×業種の構成")
                            heat = ind_rows.set_index("営業部")
                            if not heat.empty:
                                fig_heat = go.Figure(
                                    data=go.Heatmap(
                                        z=heat.to_numpy(),
                                        x=heat.columns.tolist(),
                                        y=heat.index.tolist(),
                                        colorscale="Blues",
                                        hovertemplate="営業部=%{y}<br>業種=%{x}<br>件数=%{z}<extra></extra>",
                                    )
                                )
                                fig_heat.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
                                st.plotly_chart(fig_heat, width='stretch', use_container_width=True)
                    if item.get("kind") == "industry_monitor" and item.get("rows"):
                        import pandas as pd
                        rows = item["rows"]
                        df = pd.DataFrame(rows)
                        if not df.empty:
                            display_df = df.rename(columns={
                                "industry": "業種",
                                "n": "件数",
                                "auc": "AUC",
                                "bench_ind_gap_mean": "bench-ind平均差分",
                                "bench_ind_gap_median": "bench-ind中央値",
                                "bench_ind_gap_p90": "bench-indP90",
                                "bench_ind_gap_signed_mean": "bench-ind符号平均",
                            })
                            st.dataframe(display_df, width='stretch', hide_index=True)
                    if item.get("recommendation"):
                        st.caption(f"→ {item['recommendation']}")
