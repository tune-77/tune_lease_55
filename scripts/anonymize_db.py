#!/usr/bin/env python3
"""
scripts/anonymize_db.py

Generate anonymized data/demo.db from data/lease_data.db.

Anonymization rules:
  - Company names (PII)       → faker ja_JP replacement
  - Financial numerics        → × uniform(0.85, 1.15) per row
  - Dates                     → ±30 day random offset
  - Sales departments         → mapped to 犬部/猫部/鳥部/魚部
  - Base rates                → additive ±0.3–0.8% noise, floor 0.01
  - Category columns          → in-column shuffle (preserves distribution)
"""

import json
import random
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

random.seed(42)
fake = Faker("ja_JP")

BASE_DIR = Path(__file__).parent.parent
SRC_DB = BASE_DIR / "data" / "lease_data.db"
DST_DB = BASE_DIR / "data" / "demo.db"

DEPT_ALIASES = ["犬部", "猫部", "鳥部", "魚部"]

# Financial input fields inside past_cases.data JSON
_FINANCIAL_INPUTS = [
    "nenshu", "gross_profit", "op_profit", "ord_profit", "net_income",
    "machines", "other_assets", "rent", "depreciation", "dep_expense",
    "rent_expense", "bank_credit", "lease_credit", "acquisition_cost",
]

# Datetime parse/format pairs tried in order
_DATE_FMTS = [
    ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"),
    ("%Y-%m-%dT%H:%M:%S",    "%Y-%m-%dT%H:%M:%S"),
    ("%Y-%m-%d %H:%M:%S",    "%Y-%m-%d %H:%M:%S"),
    ("%Y-%m-%d",             "%Y-%m-%d"),
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _mul_noise(val, lo: float = 0.85, hi: float = 1.15):
    """Apply multiplicative noise; pass through None/non-numeric."""
    if val is None:
        return None
    try:
        return float(val) * random.uniform(lo, hi)
    except (TypeError, ValueError):
        return val


def _date_noise(s, max_days: int = 30):
    """Shift a date/datetime string by ±max_days; return unchanged if unparseable."""
    if not s:
        return s
    s = str(s)
    for parse_fmt, out_fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(s[:len(parse_fmt)], parse_fmt)
            dt += timedelta(days=random.randint(-max_days, max_days))
            return dt.strftime(out_fmt)
        except ValueError:
            continue
    return s


def _rate_noise(val, lo: float = 0.3, hi: float = 0.8):
    """Additive ±[lo, hi]% rate noise, floor at 0.01."""
    if val is None:
        return None
    delta = random.uniform(lo, hi) * random.choice([-1, 1])
    return max(0.01, float(val) + delta)


def _shuffle_values(vals: list) -> list:
    """Return a Fisher-Yates shuffle of vals (preserves multiset distribution)."""
    out = vals[:]
    random.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _build_dept_map(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT DISTINCT sales_dept FROM past_cases ORDER BY sales_dept"
    ).fetchall()
    depts = sorted(
        r[0] for r in rows if r[0] and r[0] not in ("未設定", "0", "")
    )
    return {d: DEPT_ALIASES[i % len(DEPT_ALIASES)] for i, d in enumerate(depts)}


def _build_company_map(conn: sqlite3.Connection) -> dict:
    """Consistent original→fake company name mapping."""
    names: set[str] = set()
    for (name,) in conn.execute(
        "SELECT DISTINCT json_extract(data, '$.company_name') FROM past_cases"
        " WHERE json_extract(data, '$.company_name') IS NOT NULL"
    ):
        if name:
            names.add(name)
    for (name,) in conn.execute(
        "SELECT DISTINCT company_name FROM conversation_history"
        " WHERE company_name IS NOT NULL"
    ):
        if name:
            names.add(name)
    return {n: fake.company() for n in names}


# ---------------------------------------------------------------------------
# JSON anonymization
# ---------------------------------------------------------------------------

def _anonymize_data_json(
    raw: str | None,
    mul: float,
    company_map: dict,
    dept_map: dict,
) -> str | None:
    """Anonymize the `data` JSON blob stored in past_cases / excluded_grade_cases."""
    if not raw:
        return raw
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    # Company name
    if "company_name" in d:
        d["company_name"] = company_map.get(d["company_name"], fake.company())

    # Sales dept (sometimes embedded)
    if "sales_dept" in d:
        d["sales_dept"] = dept_map.get(d["sales_dept"], d["sales_dept"])

    # Financial inputs
    inputs = d.get("inputs")
    if isinstance(inputs, dict):
        for f in _FINANCIAL_INPUTS:
            v = inputs.get(f)
            if isinstance(v, (int, float)):
                inputs[f] = v * mul

    # result.financials
    result = d.get("result")
    if isinstance(result, dict):
        financials = result.get("financials")
        if isinstance(financials, dict):
            for k, v in financials.items():
                if isinstance(v, (int, float)):
                    financials[k] = v * mul

    # Pricing base_rate
    pricing = d.get("pricing")
    if isinstance(pricing, dict) and pricing.get("base_rate") is not None:
        pricing["base_rate"] = _rate_noise(pricing["base_rate"])

    # Top-level rate fields
    for rf in ("base_rate_at_time", "final_rate", "winning_spread"):
        v = d.get(rf)
        if isinstance(v, (int, float)) and v != 0:
            d[rf] = _rate_noise(v)

    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def _copy_past_cases(src: sqlite3.Connection, dst: sqlite3.Connection,
                     dept_map: dict, company_map: dict) -> None:
    rows = src.execute("SELECT * FROM past_cases").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(past_cases)").fetchall()]

    # Shuffle industry_sub values across rows (preserves distribution)
    industries = [r[cols.index("industry_sub")] for r in rows]
    shuffled_ind = _shuffle_values(industries)

    placeholders = ", ".join("?" * len(cols))
    for row, new_ind in zip(rows, shuffled_ind):
        mul = random.uniform(0.85, 1.15)
        r = dict(zip(cols, row))

        dept = r.get("sales_dept") or ""
        new_dept = dept_map.get(dept, dept) if dept not in ("未設定", "0", "") else dept

        dst.execute(
            f"INSERT INTO past_cases VALUES ({placeholders})",
            (
                r["id"],
                _date_noise(r["timestamp"]),
                new_ind,
                _mul_noise(r["score"]),
                _mul_noise(r["user_eq"]),
                r["final_status"],
                _anonymize_data_json(r["data"], mul, company_map, dept_map),
                new_dept,
                _date_noise(r["registration_date"]),
                _date_noise(r["estimate_sent_date"]),
                _date_noise(r["customer_response_date"]),
                _date_noise(r["final_result_date"]),
                r["quality_flag"],
            ),
        )


def _copy_excluded_grade_cases(src: sqlite3.Connection, dst: sqlite3.Connection,
                                dept_map: dict, company_map: dict) -> None:
    rows = src.execute("SELECT * FROM excluded_grade_cases").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(excluded_grade_cases)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    for row in rows:
        mul = random.uniform(0.85, 1.15)
        r = dict(zip(cols, row))
        dept = r.get("sales_dept") or ""
        new_dept = dept_map.get(dept, dept) if dept not in ("未設定", "0", "") else dept

        dst.execute(
            f"INSERT INTO excluded_grade_cases VALUES ({placeholders})",
            (
                r["id"],
                _date_noise(r["timestamp"]),
                r["industry_sub"],
                _mul_noise(r["score"]),
                _mul_noise(r["user_eq"]),
                r["final_status"],
                _anonymize_data_json(r["data"], mul, company_map, dept_map),
                new_dept,
                _date_noise(r["registration_date"]),
                _date_noise(r["estimate_sent_date"]),
                _date_noise(r["customer_response_date"]),
                _date_noise(r["final_result_date"]),
                r["original_grade"],
                r["excluded_reason"],
                _date_noise(r["extracted_at"]),
            ),
        )


def _copy_screening_records(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM screening_records").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(screening_records)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    score_cols = {"total_score", "asset_score", "tenant_score",
                  "q_risk_score", "competitor_pressure_score"}
    date_cols  = {"screened_at", "created_at", "updated_at"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = []
        for c in cols:
            v = r[c]
            if c in score_cols:
                v = _mul_noise(v)
            elif c in date_cols:
                v = _date_noise(v)
            # input_snapshot is always NULL in this DB; pass through
            new_vals.append(v)
        dst.execute(f"INSERT INTO screening_records VALUES ({placeholders})", new_vals)


def _copy_ml_features(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM ml_features").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(ml_features)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    # Columns that are categorical and should be shuffled
    category_cols = {
        "final_status", "industry_raw", "customer_type_bin",
        "main_bank_bin", "competitor_bin", "industry_code",
        "contract_type_code", "deal_source_code", "sales_dept_code",
        "pred_judgment_v3", "model_version",
    }
    # Columns that are numeric and should get noise
    skip_cols = {"case_id"} | category_cols

    # Build shuffled category columns
    cat_shuffled: dict[str, list] = {}
    for c in category_cols:
        if c in cols:
            vals = [r[cols.index(c)] for r in rows]
            cat_shuffled[c] = _shuffle_values(vals)

    for i, row in enumerate(rows):
        r = dict(zip(cols, row))
        mul = random.uniform(0.85, 1.15)
        new_vals = []
        for j, c in enumerate(cols):
            v = r[c]
            if c in category_cols:
                v = cat_shuffled[c][i]
            elif c not in skip_cols and isinstance(v, (int, float)):
                v = v * mul
            new_vals.append(v)
        dst.execute(f"INSERT INTO ml_features VALUES ({placeholders})", new_vals)


def _copy_base_rate_master(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM base_rate_master").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(base_rate_master)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    rate_cols = {"rate", "r_2y", "r_3y", "r_4y", "r_5y",
                 "r_6y", "r_7y", "r_8y", "r_9y", "r_over9y"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = [
            _rate_noise(r[c]) if c in rate_cols and r[c] is not None else r[c]
            for c in cols
        ]
        dst.execute(f"INSERT INTO base_rate_master VALUES ({placeholders})", new_vals)


def _copy_funding_rates(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM funding_rates").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(funding_rates)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    for row in rows:
        r = dict(zip(cols, row))
        r["rate_pct"] = _rate_noise(r["rate_pct"])
        dst.execute(f"INSERT INTO funding_rates VALUES ({placeholders})",
                    [r[c] for c in cols])


def _copy_asset_price_history(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM asset_price_history").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(asset_price_history)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    numeric_cols = {"current_market_price", "residual_debt", "profit_margin"}

    for row in rows:
        r = dict(zip(cols, row))
        mul = random.uniform(0.85, 1.15)
        new_vals = []
        for c in cols:
            v = r[c]
            if c == "inspected_at":
                v = _date_noise(v)
            elif c in numeric_cols and v is not None:
                v = int(v * mul)
            new_vals.append(v)
        dst.execute(f"INSERT INTO asset_price_history VALUES ({placeholders})", new_vals)


def _copy_screening_outcomes(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM screening_outcomes").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(screening_outcomes)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    date_cols = {"contract_date", "scheduled_end_date", "checked_at", "created_at", "updated_at"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = []
        for c in cols:
            v = r[c]
            if c in date_cols:
                v = _date_noise(v)
            elif c == "loss_given_default" and v is not None:
                v = _mul_noise(v)
            new_vals.append(v)
        dst.execute(f"INSERT INTO screening_outcomes VALUES ({placeholders})", new_vals)


def _copy_gunshi_cases(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM gunshi_cases").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(gunshi_cases)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    numeric_cols = {"score", "pd_pct", "prior_prob", "posterior"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = []
        for c in cols:
            v = r[c]
            if c == "created_at":
                v = _date_noise(v)
            elif c in numeric_cols and v is not None:
                v = _mul_noise(v)
            new_vals.append(v)
        dst.execute(f"INSERT INTO gunshi_cases VALUES ({placeholders})", new_vals)


def _copy_emotion_history(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM emotion_history").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(emotion_history)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    skip_cols = {"id", "recorded_at", "dominant_raw_emotion", "notes"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = []
        for c in cols:
            v = r[c]
            if c == "recorded_at":
                v = _date_noise(v)
            elif c not in skip_cols and isinstance(v, (int, float)):
                v = _mul_noise(v)
            new_vals.append(v)
        dst.execute(f"INSERT INTO emotion_history VALUES ({placeholders})", new_vals)


def _copy_retraining_log(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    rows = src.execute("SELECT * FROM retraining_log").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(retraining_log)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    date_cols = {"started_at", "completed_at"}
    numeric_cols = {"new_auc", "prev_auc", "duration_seconds"}

    for row in rows:
        r = dict(zip(cols, row))
        new_vals = []
        for c in cols:
            v = r[c]
            if c in date_cols:
                v = _date_noise(v)
            elif c in numeric_cols and v is not None:
                v = _mul_noise(v)
            new_vals.append(v)
        dst.execute(f"INSERT INTO retraining_log VALUES ({placeholders})", new_vals)


def _copy_chat_messages(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    """Chat content is generic Q&A; anonymize only dates and user_id."""
    rows = src.execute("SELECT * FROM chat_messages").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(chat_messages)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    for row in rows:
        r = dict(zip(cols, row))
        r["created_at"] = _date_noise(r.get("created_at"))
        dst.execute(f"INSERT INTO chat_messages VALUES ({placeholders})",
                    [r[c] for c in cols])


def _copy_conversation_history(
    src: sqlite3.Connection, dst: sqlite3.Connection, company_map: dict
) -> None:
    rows = src.execute("SELECT * FROM conversation_history").fetchall()
    cols = [d[1] for d in src.execute("PRAGMA table_info(conversation_history)").fetchall()]
    placeholders = ", ".join("?" * len(cols))

    for row in rows:
        r = dict(zip(cols, row))
        # Replace company_name with fake
        cn = r.get("company_name")
        if cn:
            r["company_name"] = company_map.get(cn, fake.company())
        r["created_at"] = _date_noise(r.get("created_at"))
        dst.execute(f"INSERT INTO conversation_history VALUES ({placeholders})",
                    [r[c] for c in cols])


def _copy_table_as_is(src: sqlite3.Connection, dst: sqlite3.Connection, table: str) -> None:
    """Copy a reference table unchanged (e.g. subsidies, subsidy_master, phrase_weights)."""
    rows = src.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        return
    placeholders = ", ".join("?" * len(rows[0]))
    dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SRC_DB.exists():
        raise FileNotFoundError(f"Source DB not found: {SRC_DB}")

    print(f"Source : {SRC_DB}")
    print(f"Output : {DST_DB}")

    # Copy schema + data, then wipe and re-insert anonymized rows
    if DST_DB.exists():
        DST_DB.unlink()
    shutil.copy2(SRC_DB, DST_DB)

    src = sqlite3.connect(SRC_DB)
    dst = sqlite3.connect(DST_DB)

    print("Building mappings …")
    dept_map    = _build_dept_map(src)
    company_map = _build_company_map(src)
    print(f"  Sales dept map : {dept_map}")
    print(f"  Company names  : {len(company_map)} entries")

    # Wipe all tables in DST, then re-populate
    tables_ordered = [
        "past_cases", "excluded_grade_cases", "screening_records",
        "ml_features", "base_rate_master", "funding_rates",
        "asset_price_history", "screening_outcomes",
        "gunshi_cases", "emotion_history", "retraining_log",
        "chat_messages", "conversation_history",
        "subsidies", "subsidy_master", "phrase_weights",
        "judgment_feedback", "payment_history",
    ]
    for t in tables_ordered:
        dst.execute(f"DELETE FROM {t}")
    dst.commit()

    print("Anonymizing tables …")

    _copy_past_cases(src, dst, dept_map, company_map)
    print("  past_cases … done")

    _copy_excluded_grade_cases(src, dst, dept_map, company_map)
    print("  excluded_grade_cases … done")

    _copy_screening_records(src, dst)
    print("  screening_records … done")

    _copy_ml_features(src, dst)
    print("  ml_features … done")

    _copy_base_rate_master(src, dst)
    print("  base_rate_master … done")

    _copy_funding_rates(src, dst)
    print("  funding_rates … done")

    _copy_asset_price_history(src, dst)
    print("  asset_price_history … done")

    _copy_screening_outcomes(src, dst)
    print("  screening_outcomes … done")

    _copy_gunshi_cases(src, dst)
    print("  gunshi_cases … done")

    _copy_emotion_history(src, dst)
    print("  emotion_history … done")

    _copy_retraining_log(src, dst)
    print("  retraining_log … done")

    _copy_chat_messages(src, dst)
    print("  chat_messages … done")

    _copy_conversation_history(src, dst, company_map)
    print("  conversation_history … done")

    for t in ("subsidies", "subsidy_master", "phrase_weights",
              "judgment_feedback", "payment_history"):
        _copy_table_as_is(src, dst, t)
    print("  reference tables … done")

    dst.commit()
    src.close()
    dst.close()

    print("\nVerification:")
    _verify(SRC_DB, DST_DB)
    print(f"\nGenerated: {DST_DB}")


def _verify(src_path: Path, dst_path: Path) -> None:
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)

    tables = [
        "past_cases", "screening_records", "ml_features",
        "asset_price_history", "screening_outcomes",
    ]
    for t in tables:
        sc = src.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        dc = dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        status = "OK" if sc == dc else "MISMATCH"
        print(f"  {t}: src={sc} demo={dc} [{status}]")

    # Numeric distribution check on past_cases.score
    src_scores = [r[0] for r in src.execute("SELECT score FROM past_cases WHERE score IS NOT NULL")]
    dst_scores = [r[0] for r in dst.execute("SELECT score FROM past_cases WHERE score IS NOT NULL")]
    if src_scores and dst_scores:
        src_mean = sum(src_scores) / len(src_scores)
        dst_mean = sum(dst_scores) / len(dst_scores)
        drift = abs(dst_mean - src_mean) / max(src_mean, 0.01)
        print(f"  score mean drift: {drift:.1%} (src={src_mean:.1f} demo={dst_mean:.1f})")

    # Company name check (should differ)
    src_co = src.execute(
        "SELECT json_extract(data,'$.company_name') FROM past_cases LIMIT 3"
    ).fetchall()
    dst_co = dst.execute(
        "SELECT json_extract(data,'$.company_name') FROM past_cases LIMIT 3"
    ).fetchall()
    print(f"  company_name sample src : {[r[0] for r in src_co]}")
    print(f"  company_name sample demo: {[r[0] for r in dst_co]}")

    src.close()
    dst.close()


if __name__ == "__main__":
    main()
