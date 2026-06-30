"""
案件・係数・重みの読み書きモジュール（lease_logic_sumaho10）
load_all_cases, save_all_cases, save_case_log, load_coeff_overrides, save_coeff_overrides,
get_score_weights, get_model_blend_weights, get_effective_coeffs, load_consultation_memory, append_consultation_memory,
load_case_news, append_case_news, find_similar_past_cases を提供。
st は使わず、保存失敗時は False/None を返す。呼び元で st.error 等を表示すること。
"""
import os
import sys
import json
import datetime
from typing import Optional
import numpy as np
import pandas as pd
from runtime_paths import ensure_cloudrun_demo_db_seeded, get_data_dir

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from coeff_definitions import COEFFS
from charts import _equity_ratio_display

# ローカルではリポジトリ内 data/、Cloud Run等では DATA_DIR で差し替える。
_DATA_DIR = str(get_data_dir())
CASES_FILE = os.path.join(os.path.dirname(_DATA_DIR), "past_cases.jsonl") # obsolete
DB_PATH = os.path.join(_DATA_DIR, "lease_data.db")
COEFF_OVERRIDES_FILE = os.path.join(_DATA_DIR, "coeff_overrides.json")
COEFF_AUTO_FILE      = os.path.join(_DATA_DIR, "coeff_auto.json")
COEFF_HISTORY_FILE   = os.path.join(_DATA_DIR, "coeff_history.jsonl")
CONSULTATION_MEMORY_FILE = os.path.join(_DATA_DIR, "consultation_memory.jsonl")
CASE_NEWS_FILE = os.path.join(_DATA_DIR, "case_news.jsonl")
DASHBOARD_STATS_CACHE_FILE = os.path.join(_DATA_DIR, "dashboard_stats_cache.json")
DEPARTMENT_STATS_CACHE_FILE = os.path.join(_DATA_DIR, "department_stats_cache.json")

import hashlib
import base64
import sqlite3
from contextlib import closing, contextmanager


def _open_db(path: str = DB_PATH):
    """WAL + busy_timeout を設定した SQLite 接続を返す共通ヘルパ。"""
    ensure_cloudrun_demo_db_seeded()
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _cloud_db_enabled() -> bool:
    """Cloud Run/Cloud SQL mode when DATABASE_URL is configured."""
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _db_placeholder() -> str:
    return "%s" if _cloud_db_enabled() else "?"


@contextmanager
def _case_db_connection():
    """Open the case DB against Cloud SQL when configured, otherwise SQLite."""
    if _cloud_db_enabled():
        from api.db_connection import ensure_schema, get_connection

        ensure_schema()
        with get_connection() as conn:
            yield conn
    else:
        with closing(_open_db()) as conn:
            yield conn


def hash_company_no(co_no: str) -> str:
    """企業番号（6桁数字など）を6文字の不可逆な英数字ハッシュに変換する"""
    if not co_no:
        return ""
    # すでにハッシュ化済み（6文字の英数字だが、数字だけではない）の場合はそのまま返す
    if len(co_no) == 6 and co_no.isalnum() and not co_no.isdigit():
        return co_no
    salted = f"{co_no}_tune_lease_secure_salt"
    digest = hashlib.sha256(salted.encode()).digest()
    b64 = base64.urlsafe_b64encode(digest).decode().replace("_", "").replace("-", "")
    return b64[:6].upper()

# スコア重みのデフォルト（借手/物件、総合/定性）。回帰最適化で上書き可能。
DEFAULT_WEIGHT_BORROWER = 0.85
DEFAULT_WEIGHT_ASSET = 0.15
DEFAULT_WEIGHT_QUANT = 0.6
DEFAULT_WEIGHT_QUAL = 0.4


def _normalize_rate_value(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    if v > 1000:
        return v / 1000.0
    if v > 100:
        return v / 100.0
    if v < 0.1:
        return v * 100.0
    return v


def _case_month_for_base_rate(data: dict) -> str | None:
    for key in ("final_result_date", "registration_date", "timestamp"):
        raw = str(data.get(key) or "")
        if len(raw) >= 7 and raw[:4].isdigit() and raw[4] in ("-", "/"):
            return raw[:7].replace("/", "-")
    return None


def _infer_base_rate_at_time(data: dict) -> float:
    current = _normalize_rate_value(data.get("base_rate_at_time"))
    if current > 0:
        return current
    try:
        from base_rate_master import get_base_rate_by_term

        inputs = data.get("inputs") or {}
        lease_term = int(inputs.get("lease_term") or data.get("lease_term") or 60)
        inferred = get_base_rate_by_term(month=_case_month_for_base_rate(data), lease_term_months=lease_term)
        return float(inferred) if inferred is not None else 0.0
    except Exception:
        return 0.0


def _enrich_rate_fields(data: dict) -> None:
    """保存時に基準金利を補完し、獲得スプレッドを再計算する。"""
    base_rate = _infer_base_rate_at_time(data)
    if base_rate > 0:
        data["base_rate_at_time"] = base_rate

    final_rate = _normalize_rate_value(data.get("final_rate"))
    if final_rate > 0 and base_rate > 0:
        data["final_rate"] = final_rate
        data["winning_spread"] = round(final_rate - base_rate, 6)


def load_consultation_memory(max_entries=20):
    """AI審査オフィサー相談のメモを読み込む。直近 max_entries 件を返す。"""
    if not os.path.exists(CONSULTATION_MEMORY_FILE):
        return []
    entries = []
    try:
        with open(CONSULTATION_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
    except (OSError, IOError, PermissionError):
        return []
    return entries[-max_entries:] if len(entries) > max_entries else entries


def append_consultation_memory(user_text: str, assistant_text: str):
    """相談1往復をメモに追記。失敗時は静かに無視。"""
    try:
        with open(CONSULTATION_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user": (user_text or "")[:5000],
                "assistant": (assistant_text or "")[:5000],
                "ts": datetime.datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_all_cases():
    """過去案件を全件読み込み（past_cases のみ）。
    統計用の screening_records は集計バッチ（aggregate_stats_from_past_cases.py）で別途管理。
    """
    import sqlite3

    cases = []
    if not _cloud_db_enabled() and not os.path.exists(DB_PATH):
        return cases
    try:
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM past_cases ORDER BY timestamp ASC")
            for row in cursor.fetchall():
                try:
                    d = json.loads(row[0])
                    if d.get("id"):
                        # ── 「検収」または「検収完了」は分析上「成約」として集計する ──
                        if d.get("final_status") in ("検収", "検収完了"):
                            d["final_status"] = "成約"
                        cases.append(d)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[Error in load_all_cases]: {e}", file=sys.stderr)
    return cases


def _compact_recent_case(case: dict) -> dict:
    result = case.get("result") or {}
    return {
        "timestamp": case.get("timestamp"),
        "final_status": case.get("final_status"),
        "industry_major": case.get("industry_major") or "",
        "industry_sub": case.get("industry_sub") or "",
        "result": {
            "score": result.get("score"),
            "hantei": result.get("hantei"),
        },
    }


def _to_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        v = float(value)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _case_result(case: dict) -> dict:
    result = case.get("result")
    return result if isinstance(result, dict) else {}


def _case_inputs(case: dict) -> dict:
    inputs = case.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _case_sales_dept(case: dict) -> str:
    inputs = _case_inputs(case)
    dept = case.get("sales_dept") or inputs.get("sales_dept") or "未設定"
    return str(dept).strip() or "未設定"


def _is_valid_department(dept: str) -> bool:
    normalized = str(dept or "").strip()
    return normalized not in {"", "未設定", "0", "０", "-", "不明"}


def _case_industry_major(case: dict) -> str:
    inputs = _case_inputs(case)
    major = case.get("industry_major") or inputs.get("industry_major") or "不明"
    return str(major).strip() or "不明"


def _case_score_for_department(case: dict) -> float | None:
    result = _case_result(case)
    for value in (
        result.get("score_borrower"),
        case.get("score_borrower"),
        result.get("score"),
        case.get("score"),
    ):
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _case_rate_for_department(case: dict) -> float | None:
    for key in ("final_rate", "contract_rate", "rate"):
        parsed = _to_float(case.get(key))
        if parsed is not None and parsed > 0:
            return _normalize_rate_value(parsed)
    return None


def _case_spread_for_department(case: dict) -> float | None:
    parsed = _to_float(case.get("winning_spread"))
    if parsed is not None:
        return parsed
    final_rate = _case_rate_for_department(case)
    base_rate = _to_float(case.get("base_rate_at_time"))
    if final_rate is None or base_rate is None or base_rate <= 0:
        return None
    return final_rate - _normalize_rate_value(base_rate)


def _case_contract_amount_for_department(case: dict) -> float | None:
    inputs = _case_inputs(case)
    for value in (
        case.get("contract_amount"),
        case.get("final_amount"),
        case.get("acquisition_cost"),
        inputs.get("acquisition_cost"),
    ):
        parsed = _to_float(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _case_month_for_department(case: dict) -> str | None:
    for key in ("final_result_date", "registration_date", "timestamp"):
        raw = str(case.get(key) or "")
        if len(raw) >= 7 and raw[:4].isdigit() and raw[4] in ("-", "/"):
            return raw[:7].replace("/", "-")
    return None


def _mean(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and np.isfinite(v)]
    return float(sum(cleaned) / len(cleaned)) if cleaned else None


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


def _amount_to_display_million(value: float | None) -> float | None:
    """ケース単位で M(百万円) へ正規化済みのため、ここでは恒等関数として返す。"""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def build_department_stats_cache() -> dict:
    """営業部ダッシュボード用の軽量集計を作る。"""
    all_cases = load_all_cases()
    dept_buckets: dict[str, dict] = {}
    industry_set: set[str] = set()

    for case in all_cases:
        dept = _case_sales_dept(case)
        if not _is_valid_department(dept):
            continue
        industry = _case_industry_major(case)
        status = str(case.get("final_status") or "未登録")
        score = _case_score_for_department(case)
        final_rate = _case_rate_for_department(case)
        spread = _case_spread_for_department(case)
        contract_amount = _case_contract_amount_for_department(case)
        month = _case_month_for_department(case)

        bucket = dept_buckets.setdefault(
            dept,
            {
                "department": dept,
                "total_count": 0,
                "won_count": 0,
                "lost_count": 0,
                "pending_count": 0,
                "scores": [],
                "rates": [],
                "spreads": [],
                "contract_amounts": [],
                "industries": {},
            },
        )
        bucket["total_count"] += 1
        if status == "成約":
            bucket["won_count"] += 1
        elif status == "失注":
            bucket["lost_count"] += 1
        else:
            bucket["pending_count"] += 1
        if score is not None:
            bucket["scores"].append(score)
        if final_rate is not None:
            bucket["rates"].append(final_rate)
        if spread is not None:
            bucket["spreads"].append(spread)
        if status == "成約" and contract_amount is not None:
            bucket["contract_amounts"].append(contract_amount)
        bucket["industries"][industry] = bucket["industries"].get(industry, 0) + 1
        industry_set.add(industry)

    departments = []
    for bucket in dept_buckets.values():
        decided = bucket["won_count"] + bucket["lost_count"]
        avg_score = _mean(bucket["scores"])
        avg_rate = _mean(bucket["rates"])
        avg_spread = _mean(bucket["spreads"])
        top_industry = None
        if bucket["industries"]:
            top_industry = max(bucket["industries"].items(), key=lambda item: item[1])[0]
        departments.append(
            {
                "department": bucket["department"],
                "total_count": bucket["total_count"],
                "won_count": bucket["won_count"],
                "lost_count": bucket["lost_count"],
                "pending_count": bucket["pending_count"],
                "decided_count": decided,
                "contract_rate": _round_or_none((bucket["won_count"] / decided) * 100.0 if decided else None, 1),
                "avg_score": _round_or_none(avg_score, 1),
                "avg_rate": _round_or_none(avg_rate, 2),
                "avg_spread": _round_or_none(avg_spread, 2),
                "top_industry": top_industry,
                "industries": bucket["industries"],
            }
        )

    overall_decided = sum(d["decided_count"] for d in departments)
    overall_won = sum(d["won_count"] for d in departments)
    all_scores = [score for b in dept_buckets.values() for score in b["scores"]]
    all_rates = [rate for b in dept_buckets.values() for rate in b["rates"]]
    all_spreads = [spread for b in dept_buckets.values() for spread in b["spreads"]]
    all_amounts = [amount for b in dept_buckets.values() for amount in b["contract_amounts"]]

    overall = {
        "total_count": sum(d["total_count"] for d in departments),
        "won_count": overall_won,
        "lost_count": sum(d["lost_count"] for d in departments),
        "pending_count": sum(d["pending_count"] for d in departments),
        "decided_count": overall_decided,
        "contract_rate": _round_or_none((overall_won / overall_decided) * 100.0 if overall_decided else None, 1),
        "avg_score": _round_or_none(_mean(all_scores), 1),
        "avg_rate": _round_or_none(_mean(all_rates), 2),
        "avg_spread": _round_or_none(_mean(all_spreads), 2),
        "avg_contract_amount": _round_or_none(_mean(all_amounts), 0),
        "avg_contract_amount_million": _round_or_none(_amount_to_display_million(_mean(all_amounts)), 2) if all_amounts else None,
        "total_contract_amount": _round_or_none(sum(all_amounts), 0) if all_amounts else None,
        "total_contract_amount_million": _round_or_none(_amount_to_display_million(sum(all_amounts)), 2) if all_amounts else None,
    }

    contract_rank = sorted(
        departments,
        key=lambda d: (d["contract_rate"] is not None, d["contract_rate"] or -1, d["decided_count"]),
        reverse=True,
    )
    score_rank = sorted(
        departments,
        key=lambda d: (d["avg_score"] is not None, d["avg_score"] or -1, d["total_count"]),
        reverse=True,
    )
    rate_rank = sorted(
        departments,
        key=lambda d: (d["avg_rate"] is not None, -(d["avg_rate"] or 0), d["total_count"]),
        reverse=True,
    )

    contract_rank_map = {d["department"]: i + 1 for i, d in enumerate(contract_rank)}
    score_rank_map = {d["department"]: i + 1 for i, d in enumerate(score_rank)}
    rate_rank_map = {d["department"]: i + 1 for i, d in enumerate(rate_rank)}

    for d in departments:
        d["contract_rate_diff"] = _round_or_none(
            d["contract_rate"] - overall["contract_rate"]
            if d["contract_rate"] is not None and overall["contract_rate"] is not None
            else None,
            1,
        )
        d["avg_score_diff"] = _round_or_none(
            d["avg_score"] - overall["avg_score"]
            if d["avg_score"] is not None and overall["avg_score"] is not None
            else None,
            1,
        )
        d["avg_rate_diff"] = _round_or_none(
            d["avg_rate"] - overall["avg_rate"]
            if d["avg_rate"] is not None and overall["avg_rate"] is not None
            else None,
            2,
        )
        d["contract_rate_rank"] = contract_rank_map.get(d["department"])
        d["avg_score_rank"] = score_rank_map.get(d["department"])
        d["avg_rate_rank"] = rate_rank_map.get(d["department"])

    departments.sort(key=lambda d: (d["total_count"], d["department"]), reverse=True)

    top_industries = sorted(
        industry_set,
        key=lambda ind: sum((d.get("industries") or {}).get(ind, 0) for d in departments),
        reverse=True,
    )[:8]
    industry_composition = []
    for d in departments:
        row = {"department": d["department"]}
        row.update({industry: (d.get("industries") or {}).get(industry, 0) for industry in top_industries})
        industry_composition.append(row)

    industry_metric_buckets: dict[tuple[str, str], dict] = {}
    monthly_metric_buckets: dict[tuple[str, str], dict] = {}
    for case in all_cases:
        dept = _case_sales_dept(case)
        if not _is_valid_department(dept):
            continue
        industry = _case_industry_major(case)
        final_rate = _case_rate_for_department(case)
        amount = _case_contract_amount_for_department(case)
        month = _case_month_for_department(case)
        is_won = str(case.get("final_status") or "") == "成約"

        industry_bucket = industry_metric_buckets.setdefault(
            (dept, industry),
            {"department": dept, "industry": industry, "rates": [], "amounts": [], "count": 0, "won_count": 0},
        )
        industry_bucket["count"] += 1
        if final_rate is not None:
            industry_bucket["rates"].append(final_rate)
        if is_won and amount is not None:
            industry_bucket["amounts"].append(amount)
            industry_bucket["won_count"] += 1

        if month:
            monthly_bucket = monthly_metric_buckets.setdefault(
                (dept, month),
                {"department": dept, "month": month, "rates": [], "amounts": [], "count": 0, "won_count": 0},
            )
            monthly_bucket["count"] += 1
            if final_rate is not None:
                monthly_bucket["rates"].append(final_rate)
            if is_won and amount is not None:
                monthly_bucket["amounts"].append(amount)
                monthly_bucket["won_count"] += 1

    industry_metrics = []
    for bucket in industry_metric_buckets.values():
        avg_amount = _mean(bucket["amounts"])
        if not bucket["rates"] and not bucket["amounts"]:
            continue
        industry_metrics.append(
            {
                "department": bucket["department"],
                "industry": bucket["industry"],
                "count": bucket["count"],
                "won_count": bucket["won_count"],
                "avg_rate": _round_or_none(_mean(bucket["rates"]), 2),
                "avg_contract_amount": _round_or_none(avg_amount, 0),
                "avg_contract_amount_million": _round_or_none(_amount_to_display_million(avg_amount), 2),
                "total_contract_amount": _round_or_none(sum(bucket["amounts"]), 0) if bucket["amounts"] else None,
                "total_contract_amount_million": _round_or_none(_amount_to_display_million(sum(bucket["amounts"])), 2) if bucket["amounts"] else None,
            }
        )
    industry_metrics.sort(key=lambda row: (row["department"], -(row["won_count"] or 0), row["industry"]))

    monthly_metrics = []
    for bucket in monthly_metric_buckets.values():
        avg_amount = _mean(bucket["amounts"])
        if not bucket["rates"] and not bucket["amounts"]:
            continue
        monthly_metrics.append(
            {
                "department": bucket["department"],
                "month": bucket["month"],
                "count": bucket["count"],
                "won_count": bucket["won_count"],
                "avg_rate": _round_or_none(_mean(bucket["rates"]), 2),
                "avg_contract_amount": _round_or_none(avg_amount, 0),
                "avg_contract_amount_million": _round_or_none(_amount_to_display_million(avg_amount), 2),
                "total_contract_amount": _round_or_none(sum(bucket["amounts"]), 0) if bucket["amounts"] else None,
                "total_contract_amount_million": _round_or_none(_amount_to_display_million(sum(bucket["amounts"])), 2) if bucket["amounts"] else None,
            }
        )
    monthly_metrics.sort(key=lambda row: (row["department"], row["month"]))

    significance_summary = []
    try:
        from model_review_hooks import _evaluate_department_significance

        sig = _evaluate_department_significance(
            {
                "id": "sales_dept_significance",
                "kind": "department_significance",
                "target": "score_borrower",
                "min_cases_per_dept": 8,
                "alpha": 0.05,
                "min_effect": 0.05,
            }
        )
        for row in sig.get("results") or []:
            significance_summary.append(
                {
                    "item": row.get("項目"),
                    "test": row.get("検定"),
                    "p_value": _round_or_none(row.get("p値"), 4),
                    "effect_size": _round_or_none(row.get("効果量"), 3),
                    "significance": row.get("有意"),
                    "note": row.get("補足"),
                }
            )
    except Exception:
        significance_summary = []

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "overall": overall,
        "departments": departments,
        "industry_keys": top_industries,
        "industry_composition": industry_composition,
        "industry_metrics": industry_metrics,
        "monthly_metrics": monthly_metrics,
        "significance_summary": significance_summary[:8],
    }


def build_dashboard_stats_cache(limit_recent_cases: int = 15) -> dict:
    """ホーム画面用の軽量集計を作る。"""
    try:
        from analysis_regression import run_contract_driver_analysis
        analysis = run_contract_driver_analysis() or {}
    except Exception:
        analysis = {}

    all_cases = load_all_cases()
    recent_cases = [_compact_recent_case(c) for c in reversed(all_cases[-limit_recent_cases:])] if all_cases else []

    closed_cases = analysis.get("closed_cases") or []
    scores = []
    for c in closed_cases:
        res = c.get("result") or {}
        score_borrower = res.get("score_borrower")
        if isinstance(score_borrower, (int, float)):
            scores.append(float(score_borrower))

    avg_score_borrower = sum(scores) / len(scores) if scores else None

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "analysis": {
            "closed_count": analysis.get("closed_count"),
            "avg_financials": analysis.get("avg_financials"),
            "tag_ranking": analysis.get("tag_ranking"),
            "top3_drivers": analysis.get("top3_drivers"),
            "qualitative_summary": analysis.get("qualitative_summary"),
            "avg_score_borrower": avg_score_borrower,
        },
        "recent_cases": recent_cases,
    }


def load_dashboard_stats_cache() -> dict | None:
    if not os.path.exists(DASHBOARD_STATS_CACHE_FILE):
        return None
    try:
        with open(DASHBOARD_STATS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def refresh_dashboard_stats_cache() -> dict | None:
    payload = build_dashboard_stats_cache()
    try:
        os.makedirs(os.path.dirname(DASHBOARD_STATS_CACHE_FILE), exist_ok=True)
        tmp_path = DASHBOARD_STATS_CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, cls=CustomJSONEncoder)
        os.replace(tmp_path, DASHBOARD_STATS_CACHE_FILE)
    except Exception as e:
        print(f"[Error in refresh_dashboard_stats_cache]: {e}", file=sys.stderr)
    return payload


def load_department_stats_cache() -> dict | None:
    if not os.path.exists(DEPARTMENT_STATS_CACHE_FILE):
        return None
    try:
        with open(DEPARTMENT_STATS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def refresh_department_stats_cache() -> dict | None:
    payload = build_department_stats_cache()
    try:
        os.makedirs(os.path.dirname(DEPARTMENT_STATS_CACHE_FILE), exist_ok=True)
        tmp_path = DEPARTMENT_STATS_CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, cls=CustomJSONEncoder)
        os.replace(tmp_path, DEPARTMENT_STATS_CACHE_FILE)
    except Exception as e:
        print(f"[Error in refresh_department_stats_cache]: {e}", file=sys.stderr)
    return payload


def refresh_stats_caches() -> None:
    refresh_dashboard_stats_cache()
    refresh_department_stats_cache()


def load_past_cases():
    """save_case_log で保存された過去の審査ログをすべて読み込む。"""
    return load_all_cases()


try:
    import streamlit as _st

    @_st.cache_data(ttl=60)
    def load_all_cases_cached() -> list:
        """load_all_cases の 1 分キャッシュ版（Streamlit 環境専用）。"""
        return load_all_cases()

except Exception:
    def load_all_cases_cached() -> list:  # type: ignore[misc]
        return load_all_cases()


def find_similar_past_cases(current_case_data: dict, max_count: int = 3):
    """
    現在の案件データに基づき、財務・属性の近い過去案件を高度な手法で検索する。
    """
    all_past = load_all_cases()
    if not all_past:
        return []

    from case_similarity import CaseSimilarityEngine
    engine = CaseSimilarityEngine(all_past)
    
    similar_results = engine.find_similar(current_case_data, top_n=max_count)
    
    # UI表示用に必要な情報を抽出して返す
    output = []
    for item in similar_results:
        case = item["case"]
        output.append({
            "id": case.get("id"),
            "name": case.get("borrower_name", "匿名企業"),
            "industry": case.get("industry_sub", ""),
            "score": item["case"].get("result", {}).get("score", 0),
            "status": case.get("final_status", "未登録"),
            "similarity": round(item["similarity"] * 100, 1),
            "equity": round(float(case.get("equity_ratio", 0) or case.get("user_eq", 0) or 0) * 100, 1),
            "revenue": case.get("nenshu", 0),
            "result": case.get("result", {}),
            "data": case
        })
    return output


def analyze_lost_cases(industry_sub=None):
    """
    失注案件の統計を分析する。
    """
    all_cases = load_all_cases()
    # ケースの data プロパティまたはルートからステータスを確認
    lost_cases = []
    for c in all_cases:
        status = c.get("final_status")
        if status == "失注":
            lost_cases.append(c)
    
    if industry_sub:
        lost_cases = [c for c in lost_cases if c.get("industry_sub") == industry_sub]
        
    reasons = {}
    competitors = {}
    comp_rates = []
    
    for c in lost_cases:
        r = c.get("lost_reason", "不明")
        if not r: r = "不明"
        reasons[r] = reasons.get(r, 0) + 1
        
        comp = c.get("competitor_name", "不明")
        if comp:
            competitors[comp] = competitors.get(comp, 0) + 1
        
        rate = c.get("competitor_rate")
        if rate and isinstance(rate, (int, float)):
            comp_rates.append(rate)
            
    return {
        "total": len(lost_cases),
        "reasons": reasons,
        "competitors": competitors,
        "avg_competitor_rate": sum(comp_rates) / len(comp_rates) if comp_rates else None,
        "cases": lost_cases
    }

def delete_case(case_id: str) -> bool:
    """指定IDの案件を1件削除する。全件置き換えを使わない安全な単体削除。"""
    if not _cloud_db_enabled() and not os.path.exists(DB_PATH):
        return False
    try:
        ph = _db_placeholder()
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM past_cases WHERE id = {ph}", (case_id,))
            conn.commit()
        refresh_stats_caches()
        return True
    except Exception:
        return False


def update_case(case_id: str, updates: dict) -> bool:
    """指定IDの案件の data フィールドを更新する。全件置き換えを使わない安全な単体更新。"""
    if not _cloud_db_enabled() and not os.path.exists(DB_PATH):
        return False
    try:
        ph = _db_placeholder()
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT data FROM past_cases WHERE id = {ph}", (case_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return False
            data = json.loads(row[0])
            data.update(updates)
            final_status = data.get("final_status", "")
            json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
            cursor.execute(
                f"UPDATE past_cases SET data = {ph}, final_status = {ph} WHERE id = {ph}",
                (json_str, final_status, case_id),
            )
            conn.commit()
        refresh_stats_caches()
        return True
    except Exception:
        return False


def save_all_cases(cases):
    """案件一覧をUPSERT保存。既存データは上書き、新規データは追加。既存レコードは削除しない。

    ⚠️ 注意: 後方互換のため残しているが、新規コードでは
    delete_case() / update_case() / save_case_log() を使うこと。
    """
    if not os.path.exists(DB_PATH):
        return False

    try:
        with closing(_open_db()) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN")
                # DELETEはせず、UPSERT（INSERT OR REPLACE）で処理するよう修正
                for data in cases:
                    case_id = data.get("id")
                    timestamp = data.get("timestamp", "")
                    industry_sub = data.get("industry_sub", "")
                    final_status = data.get("final_status", "")

                    score, user_eq = None, None
                    res = data.get("result", {})
                    if isinstance(res, dict):
                        score = res.get("score")
                        user_eq = res.get("user_eq")

                    try:
                        score_val = float(score) if score is not None else None
                    except (TypeError, ValueError):
                        score_val = None

                    try:
                        user_eq_val = float(user_eq) if user_eq is not None else None
                    except (TypeError, ValueError):
                        user_eq_val = None

                    json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
                    # INSERT OR REPLACE: 同一IDが存在すれば上書き、なければ追加
                    cursor.execute("""
                        INSERT OR REPLACE INTO past_cases
                        (id, timestamp, industry_sub, score, user_eq, final_status, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (case_id, timestamp, industry_sub, score_val, user_eq_val, final_status, json_str))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return True
    except Exception:
        return False


def load_coeff_overrides():
    """保存済みの係数オーバーライド（手動設定）を読み込む。無ければ None。"""
    if not os.path.exists(COEFF_OVERRIDES_FILE):
        return None
    try:
        with open(COEFF_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_coeff_overrides(overrides_dict, comment: str = ""):
    """係数オーバーライド（手動設定）を JSON で保存。変更履歴も記録。失敗時は False。"""
    dirpath = os.path.dirname(COEFF_OVERRIDES_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        # 変更前の値を取得して差分を記録
        before = load_coeff_overrides() or {}
        with open(COEFF_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides_dict, f, ensure_ascii=False, indent=2)
        _append_coeff_history(
            change_type="manual",
            before=before,
            after=overrides_dict,
            comment=comment,
        )
        _save_governance_snapshot(overrides=overrides_dict, comment=comment)
        return True
    except Exception:
        return False


def _save_governance_snapshot(overrides: dict, comment: str = "") -> None:
    """係数オーバーライドのスナップショットを governance_snapshots.json に追記する（最大50件）。"""
    snap_path = os.path.join(_DATA_DIR, "governance_snapshots.json")
    try:
        if os.path.exists(snap_path):
            with open(snap_path, "r", encoding="utf-8") as f:
                snaps = json.load(f)
        else:
            snaps = []
        snaps.append({
            "id": f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "comment": comment,
            "overrides": overrides,
        })
        snaps = snaps[-50:]
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snaps, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_governance_snapshots() -> list:
    """ガバナンス・スナップショット一覧を返す（新しい順）。無ければ空リスト。"""
    snap_path = os.path.join(_DATA_DIR, "governance_snapshots.json")
    try:
        if not os.path.exists(snap_path):
            return []
        with open(snap_path, "r", encoding="utf-8") as f:
            snaps = json.load(f)
        return list(reversed(snaps))
    except Exception:
        return []


def load_auto_coeffs() -> dict:
    """自動最適化で生成された推奨重みを読み込む。無ければ空 dict。"""
    if not os.path.exists(COEFF_AUTO_FILE):
        return {}
    try:
        with open(COEFF_AUTO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_auto_coeffs(auto_dict: dict, comment: str = "") -> bool:
    """自動最適化の推奨重みを専用ファイルに保存。変更履歴も記録。失敗時は False。"""
    dirpath = os.path.dirname(COEFF_AUTO_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        before = load_auto_coeffs() or {}
        with open(COEFF_AUTO_FILE, "w", encoding="utf-8") as f:
            json.dump(auto_dict, f, ensure_ascii=False, indent=2)
        _append_coeff_history(
            change_type="auto",
            before=before,
            after=auto_dict,
            comment=comment or "自動最適化による更新",
        )
        return True
    except Exception:
        return False


def _append_coeff_history(change_type: str, before: dict, after: dict, comment: str = "") -> None:
    """係数変更履歴を JSONL に1行追記する。"""
    try:
        os.makedirs(os.path.dirname(COEFF_HISTORY_FILE), exist_ok=True)
        # 変更されたキーだけ抽出
        all_keys = set(before.keys()) | set(after.keys())
        changed = {
            k: {"before": before.get(k), "after": after.get(k)}
            for k in all_keys
            if before.get(k) != after.get(k)
        }
        record = {
            "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
            "change_type": change_type,   # "manual" or "auto"
            "comment":     comment,
            "changed_keys": changed,
            "snapshot_after": after,
        }
        with open(COEFF_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 履歴書き込み失敗は本処理に影響させない


def load_coeff_history() -> list:
    """係数変更履歴を新しい順で返す。"""
    if not os.path.exists(COEFF_HISTORY_FILE):
        return []
    try:
        records = [json.loads(l) for l in open(COEFF_HISTORY_FILE, encoding="utf-8") if l.strip()]
        return list(reversed(records))
    except Exception:
        return []


def get_score_weights():
    """
    借手/物件・総合/定性の重みを返す。(w_borrower, w_asset, w_quant, w_qual)。
    優先順位: 手動設定 (coeff_overrides.json) > 自動最適化 (coeff_auto.json) > デフォルト値
    """
    auto     = load_auto_coeffs()
    manual   = load_coeff_overrides() or {}
    # score_weights キー（手動）vs _auto_weight_* キー（自動）を統合
    sw = manual.get("score_weights") or {}
    # 借手/物件重み: 手動 > 自動 > デフォルト
    w_b  = sw.get("borrower") or auto.get("_auto_weight_borrower")
    w_a  = sw.get("asset")    or auto.get("_auto_weight_asset")
    w_q  = sw.get("quant")    or auto.get("_auto_weight_quant")
    w_q2 = sw.get("qual")     or auto.get("_auto_weight_qual")
    if w_b is not None and w_a is not None and (w_b + w_a) > 0:
        s_ba = w_b + w_a
        w_borrower, w_asset = w_b / s_ba, w_a / s_ba
    else:
        w_borrower, w_asset = DEFAULT_WEIGHT_BORROWER, DEFAULT_WEIGHT_ASSET
    if w_q is not None and w_q2 is not None and (w_q + w_q2) > 0:
        s_qq = w_q + w_q2
        w_quant, w_qual = w_q / s_qq, w_q2 / s_qq
    else:
        w_quant, w_qual = DEFAULT_WEIGHT_QUANT, DEFAULT_WEIGHT_QUAL
    return (w_borrower, w_asset, w_quant, w_qual)


def get_model_blend_weights():
    """
    ① 全体モデル / ② 指標モデル / ③ 業種別モデル の混合重みを返す。
    優先順位: 手動設定 (coeff_overrides.json) > 自動最適化 (coeff_auto.json) > デフォルト (0.5/0.3/0.2)
    戻り値: (w_main, w_bench, w_ind) — 合計 1.0
    """
    _DEFAULT_MAIN  = 0.5
    _DEFAULT_BENCH = 0.3
    _DEFAULT_IND   = 0.2
    auto   = load_auto_coeffs()
    manual = load_coeff_overrides() or {}
    mw = manual.get("model_blend_weights") or {}
    w_m  = mw.get("main")  or auto.get("_auto_blend_w_main")
    w_b  = mw.get("bench") or auto.get("_auto_blend_w_bench")
    w_i  = mw.get("ind")   or auto.get("_auto_blend_w_ind")
    if w_m is not None and w_b is not None and w_i is not None:
        total = float(w_m) + float(w_b) + float(w_i)
        if total > 0:
            return float(w_m) / total, float(w_b) / total, float(w_i) / total
    return _DEFAULT_MAIN, _DEFAULT_BENCH, _DEFAULT_IND


def get_effective_coeffs(key=None):
    """指定キーの係数セットを返す。オーバーライドがあればマージ。"""
    if key is None:
        key = "全体_既存先"
    overrides = load_coeff_overrides() or {}
    base_key = key
    if base_key not in COEFFS:
        base_key = key.replace("_既存先", "").replace("_新規先", "")
    base = dict(COEFFS.get(base_key, COEFFS["全体_既存先"]))
    if overrides.get(base_key):
        base.update(overrides[base_key])
    if overrides.get(key):
        base.update(overrides[key])
    return base


def append_case_news(record: dict):
    """案件ごとのニュースを1件追記。失敗時は False。"""
    if not record:
        return True
    try:
        data = dict(record)
        data.setdefault("saved_at", datetime.datetime.now().isoformat())
        with open(CASE_NEWS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_case_news(case_id: Optional[str] = None):
    """保存済みニュースを読み込む。case_id を指定するとその案件分だけ。"""
    if not os.path.exists(CASE_NEWS_FILE):
        return []
    records = []
    try:
        with open(CASE_NEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if case_id is not None and rec.get("case_id") != case_id:
                    continue
                records.append(rec)
    except Exception:
        return []
    return records


class CustomJSONEncoder(json.JSONEncoder):
    """
    Numpyの各種数値型やPandasの欠損値等をPython標準型に変換して
    JSONシリアライズ可能にするエンコーダ。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # np.nan や pd.NA の対応
            return None
        try:
            return super().default(obj)
        except Exception:
            return str(obj)

def save_case_log(data):
    """審査1件分のログをDBに追記し、生成した案件IDを返す。失敗時は None。"""
    import uuid
    # マイクロ秒 + UUID4の先頭8桁でバッチ一括保存時のPRIMARY KEY衝突を防止
    case_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
    data["id"] = case_id
    # timestamp が呼び出し元で既にセットされている場合（バッチ履歴データ）は上書きしない
    if not data.get("timestamp"):
        data["timestamp"] = datetime.datetime.now().isoformat()
    # final_status が呼び出し元で既にセットされている場合（バッチ成約/失注）は上書きしない
    if "final_status" not in data or not data["final_status"]:
        data["final_status"] = "未登録"
    if not data.get("registration_date"):
        data["registration_date"] = str(data.get("timestamp", ""))[:10] or datetime.datetime.now().strftime("%Y-%m-%d")
    _enrich_rate_fields(data)
    
    industry_sub = data.get("industry_sub", "")
    score, user_eq = None, None
    res = data.get("result", {})
    if isinstance(res, dict):
        score, user_eq = res.get("score"), res.get("user_eq")
        
    try:
        score_val = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = None

    try:
        user_eq_val = float(user_eq) if user_eq is not None else None
    except (TypeError, ValueError):
        user_eq_val = None
        
    try:
        if not _cloud_db_enabled() and not os.path.exists(DB_PATH):
            from migrate_to_sqlite import init_db
            init_db()
            
        json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
        ph = _db_placeholder()
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            # テーブルが存在しない場合は作成（DBファイルが存在してもテーブルが欠けているケースに対応）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS past_cases (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    industry_sub TEXT,
                    score REAL,
                    user_eq REAL,
                    final_status TEXT,
                    data TEXT,
                    sales_dept TEXT DEFAULT '未設定',
                    registration_date TEXT,
                    estimate_sent_date TEXT,
                    customer_response_date TEXT,
                    final_result_date TEXT
                )
            """)
            # sales_dept カラムが存在しない古いDBへの対応
            try:
                cursor.execute("ALTER TABLE past_cases ADD COLUMN sales_dept TEXT DEFAULT '未設定'")
            except Exception:
                if _cloud_db_enabled():
                    conn.rollback()
                pass
            for col in ("registration_date", "estimate_sent_date", "customer_response_date", "final_result_date"):
                try:
                    cursor.execute(f"ALTER TABLE past_cases ADD COLUMN {col} TEXT")
                except Exception:
                    if _cloud_db_enabled():
                        conn.rollback()
                    pass
            sales_dept_val = data.get("sales_dept", "未設定") or "未設定"
            registration_date = data.get("registration_date")
            estimate_sent_date = data.get("estimate_sent_date")
            customer_response_date = data.get("customer_response_date")
            final_result_date = data.get("final_result_date")
            cursor.execute(f"""
                INSERT INTO past_cases
                (id, timestamp, industry_sub, score, user_eq, final_status, data, sales_dept,
                 registration_date, estimate_sent_date, customer_response_date, final_result_date)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            """, (
                case_id,
                data["timestamp"],
                industry_sub,
                score_val,
                user_eq_val,
                data["final_status"],
                json_str,
                sales_dept_val,
                registration_date,
                estimate_sent_date,
                customer_response_date,
                final_result_date,
            ))
            conn.commit()
        if not _cloud_db_enabled():
            _trigger_ml_features_update(case_id)
        refresh_stats_caches()
        return case_id
    except Exception as e:
        import traceback
        print(f"[Error in save_case_log]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def save_excluded_grade_case(data):
    """信用リスク群（格付8-3/9/10）を専用テーブルに保存する。失敗時は None。"""
    import uuid

    case_id = data.get("id") or datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
    data["id"] = case_id
    if not data.get("timestamp"):
        data["timestamp"] = datetime.datetime.now().isoformat()
    if "final_status" not in data or not data["final_status"]:
        data["final_status"] = "未登録"
    if not data.get("registration_date"):
        data["registration_date"] = str(data.get("timestamp", ""))[:10] or datetime.datetime.now().strftime("%Y-%m-%d")
    _enrich_rate_fields(data)

    res = data.get("result", {})
    score = res.get("score") if isinstance(res, dict) else None
    user_eq = res.get("user_eq") if isinstance(res, dict) else None
    try:
        score_val = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = None
    try:
        user_eq_val = float(user_eq) if user_eq is not None else None
    except (TypeError, ValueError):
        user_eq_val = None

    try:
        json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
        ph = _db_placeholder()
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS excluded_grade_cases (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    industry_sub TEXT,
                    score REAL,
                    user_eq REAL,
                    final_status TEXT,
                    data TEXT,
                    sales_dept TEXT,
                    registration_date TEXT,
                    estimate_sent_date TEXT,
                    customer_response_date TEXT,
                    final_result_date TEXT,
                    original_grade TEXT,
                    excluded_reason TEXT,
                    extracted_at TEXT
                )
            """)
            if _cloud_db_enabled():
                upsert_sql = f"""
                INSERT INTO excluded_grade_cases
                (id, timestamp, industry_sub, score, user_eq, final_status, data, sales_dept,
                 registration_date, estimate_sent_date, customer_response_date, final_result_date,
                 original_grade, excluded_reason, extracted_at)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                ON CONFLICT (id) DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    industry_sub = EXCLUDED.industry_sub,
                    score = EXCLUDED.score,
                    user_eq = EXCLUDED.user_eq,
                    final_status = EXCLUDED.final_status,
                    data = EXCLUDED.data,
                    sales_dept = EXCLUDED.sales_dept,
                    registration_date = EXCLUDED.registration_date,
                    estimate_sent_date = EXCLUDED.estimate_sent_date,
                    customer_response_date = EXCLUDED.customer_response_date,
                    final_result_date = EXCLUDED.final_result_date,
                    original_grade = EXCLUDED.original_grade,
                    excluded_reason = EXCLUDED.excluded_reason,
                    extracted_at = EXCLUDED.extracted_at
                """
            else:
                upsert_sql = f"""
                INSERT OR REPLACE INTO excluded_grade_cases
                (id, timestamp, industry_sub, score, user_eq, final_status, data, sales_dept,
                 registration_date, estimate_sent_date, customer_response_date, final_result_date,
                 original_grade, excluded_reason, extracted_at)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                """
            cursor.execute(upsert_sql, (
                case_id,
                data["timestamp"],
                data.get("industry_sub", ""),
                score_val,
                user_eq_val,
                data["final_status"],
                json_str,
                data.get("sales_dept", "未設定") or "未設定",
                data.get("registration_date"),
                data.get("estimate_sent_date"),
                data.get("customer_response_date"),
                data.get("final_result_date"),
                data.get("original_grade"),
                data.get("excluded_reason", "grade_8-3_9_10"),
                data.get("extracted_at") or datetime.datetime.now().isoformat(timespec="seconds"),
            ))
            conn.commit()
        return case_id
    except Exception as e:
        import traceback
        print(f"[Error in save_excluded_grade_case]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _trigger_ml_features_update(case_id: str) -> None:
    """新規登録・更新後に ml_features を非同期で更新する。エラーは握りつぶす。"""
    try:
        import importlib.util, os
        if not os.path.exists(os.path.join(_SCRIPT_DIR, "data", "ml_rf_v3.pkl")):
            return
        script = os.path.join(_SCRIPT_DIR, "scripts", "update_ml_features.py")
        spec   = importlib.util.spec_from_file_location("update_ml_features", script)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.update_ml_features(case_ids=[case_id])
    except Exception:
        pass  # メインアプリの動作を妨げない


def update_case_field(case_id: str, key: str, value: object) -> bool:
    """指定された case_id のレコードに対して、[key] = value を追加・更新する。"""
    if not case_id or (not _cloud_db_enabled() and not os.path.exists(DB_PATH)):
        return False

    try:
        ph = _db_placeholder()
        with _case_db_connection() as conn:
            cursor = conn.cursor()
            # 該当レコードのdata(JSON文字列)を取得
            cursor.execute(f"SELECT data FROM past_cases WHERE id = {ph}", (case_id,))
            row = cursor.fetchone()
            if not row:
                return False

            data_json = row[0]
            try:
                case_data = json.loads(data_json)
            except (json.JSONDecodeError, ValueError):
                case_data = {}

            # JSON側の更新
            case_data[key] = value
            new_json_str = json.dumps(case_data, ensure_ascii=False, cls=CustomJSONEncoder)

            # もし特定のキー（ステータス等）が単独カラムにもあれば一緒に更新する
            update_cols = f"data = {ph}"
            update_args = [new_json_str]

            if key == "final_status":
                update_cols += f", final_status = {ph}"
                update_args.append(str(value))
            elif key == "industry_sub":
                update_cols += f", industry_sub = {ph}"
                update_args.append(str(value))

            update_args.append(case_id)

            cursor.execute(f"UPDATE past_cases SET {update_cols} WHERE id = {ph}", tuple(update_args))
            conn.commit()
        if not _cloud_db_enabled():
            _trigger_ml_features_update(case_id)
        if key in {
            "final_status",
            "industry_sub",
            "industry_major",
            "score",
            "result",
            "final_rate",
            "base_rate_at_time",
            "final_result_date",
            "registration_date",
            "estimate_sent_date",
            "customer_response_date",
            "qualitative_scoring",
            "qualitative_scoring_correction",
        }:
            refresh_stats_caches()
        return True
    except Exception as e:
        import traceback
        print(f"[Error in update_case_field]: {e}", file=sys.stderr)
        return False
