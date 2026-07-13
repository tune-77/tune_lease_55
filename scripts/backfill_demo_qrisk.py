#!/usr/bin/env python3
"""Backfill demo SQLite data with Q_risk fields for similarity/search use.

This is intentionally deterministic. It does not try to create a new credit
model; it adds a Q_risk discovery signal and searchable tags to existing demo
cases so AURION/Q_risk views can compare all historical rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "demo.db"


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _text(value: Any) -> str:
    return str(value or "").strip()


def _stable_jitter(key: str, span: float = 3.5) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * span


def _status_group(status: str) -> str:
    status = _text(status)
    if status in {"成約", "承認", "検収", "検収完了", "approved", "won"}:
        return "won"
    if status in {"失注", "否決", "rejected", "lost"}:
        return "lost"
    return "open"


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def compute_qrisk(payload: dict[str, Any], *, row_key: str = "") -> dict[str, Any]:
    """Return a deterministic Q_risk discovery signal for one case."""
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else payload

    score = (
        _num(result.get("score_base"))
        or _num(result.get("score"))
        or _num(payload.get("score_base"))
        or _num(payload.get("score"))
        or _num(payload.get("total_score"))
    )
    borrower_score = _num(result.get("score_borrower")) or _num(payload.get("tenant_score"))
    asset_score = _num(result.get("asset_score")) or _num(inputs.get("lease_asset_score")) or _num(payload.get("asset_score"))
    status = _text(payload.get("final_status") or payload.get("outcome") or result.get("hantei"))
    status_group = _status_group(status)

    user_op = _num(result.get("user_op"))
    bench_op = _num(result.get("bench_op"))
    user_eq = _num(result.get("user_eq")) or _num(payload.get("user_eq"))
    bench_eq = _num(result.get("bench_eq"))
    user_dscr = _num(result.get("user_dscr")) or _num(payload.get("dscr"))
    user_debt = _num(result.get("user_debt_ratio"))
    current_ratio = _num(result.get("user_current_ratio"))

    op_profit = _num(inputs.get("op_profit"))
    ord_profit = _num(inputs.get("ord_profit"))
    net_income = _num(inputs.get("net_income"))
    sales = _num(inputs.get("nenshu")) or _num(inputs.get("annual_revenue"))
    bank_credit = _num(inputs.get("bank_credit"))
    lease_credit = _num(inputs.get("lease_credit"))
    acquisition_cost = _num(inputs.get("acquisition_cost")) or _num(inputs.get("lease_amount"))

    customer_type = _text(payload.get("customer_type") or inputs.get("customer_type"))
    main_bank = _text(payload.get("main_bank") or inputs.get("main_bank"))
    competitor = _text(payload.get("competitor") or inputs.get("competitor"))
    competitor_rate = _num(payload.get("competitor_rate") or inputs.get("competitor_rate"))
    deal_source = _text(payload.get("deal_source") or inputs.get("deal_source"))
    industry_sub = _text(payload.get("industry_sub") or inputs.get("industry_sub") or result.get("industry_sub"))
    asset_name = _text(result.get("asset_name") or inputs.get("lease_asset_name") or inputs.get("asset_name"))

    q = 14.0
    reasons: list[str] = []
    tags: list[str] = []

    def add(points: float, reason: str, tag: str) -> None:
        nonlocal q
        q += points
        reasons.append(reason)
        tags.append(tag)

    if score is not None:
        if status_group == "lost" and score >= 70:
            add(26, "高スコアなのに失注しており、価格・競合・銀行支援など非スコア因子を確認", "high_score_lost")
        elif status_group == "lost" and score >= 55:
            add(14, "中位以上のスコアで失注しており、条件提示・競合差を確認", "mid_score_lost")
        elif status_group == "won" and score < 45:
            add(24, "低スコアなのに成約しており、保全・支援・個別事情を確認", "low_score_won")
        elif status_group == "won" and score < 60:
            add(12, "境界スコアで成約しており、通した条件を再利用候補として確認", "borderline_won")
        if score >= 75 and borrower_score is not None and borrower_score < 45:
            add(14, "総合スコアと借手スコアに乖離", "score_borrower_gap")
        if score >= 70 and asset_score is not None and asset_score < 45:
            add(10, "総合スコアは高いが物件保全が弱い", "asset_gap")

    if user_op is not None and bench_op is not None and user_op + 2.0 < bench_op:
        add(8 if user_op >= 0 else 14, "営業利益率が業界目安を下回る", "op_margin_below_bench")
    elif op_profit is not None and op_profit < 0:
        add(12, "営業赤字", "operating_loss")

    if any(v is not None and v < 0 for v in (ord_profit, net_income)):
        add(10, "経常/当期利益に赤字項目", "profit_loss")

    if user_eq is not None:
        if user_eq < 10:
            add(16, "自己資本比率が薄い", "thin_equity")
        elif bench_eq is not None and user_eq + 8 < bench_eq:
            add(9, "自己資本比率が業界目安を下回る", "equity_below_bench")

    if user_dscr is not None and user_dscr < 1.5:
        add(16, "DSCRが低い", "low_dscr")
    if user_debt is not None and user_debt >= 90:
        add(8, "負債比率が高い", "high_debt_ratio")
    if current_ratio is not None and current_ratio < 90:
        add(6, "流動比率が低い", "low_current_ratio")

    if customer_type == "新規先":
        add(5, "新規先", "new_customer")
    if "非メイン" in main_bank or "弱" in main_bank:
        add(7, "銀行支援が弱い/非メイン", "weak_bank_support")
    elif "メイン" in main_bank:
        q -= 3

    if "競合あり" in competitor or competitor_rate:
        add(8, "競合条件の影響あり", "competitor_pressure")
    elif "競合なし" in competitor:
        q -= 2

    if sales and acquisition_cost and acquisition_cost > max(30_000, sales * 0.25):
        add(7, "売上規模に対して投資額が重い", "large_investment")
    if bank_credit and sales and bank_credit > sales * 0.8:
        add(5, "売上規模に対して銀行与信が重い", "bank_credit_heavy")
    if lease_credit and sales and lease_credit > sales * 0.25:
        add(5, "売上規模に対してリース与信が重い", "lease_credit_heavy")

    q += _stable_jitter(row_key or json.dumps(payload, ensure_ascii=False, sort_keys=True)[:500])
    q = max(0.0, min(88.0, q))
    if not reasons:
        reasons.append("重大な矛盾は少なく、近い業種・スコア帯の通常比較に使う")
        tags.append("normal_compare")

    if q >= 60:
        level = "high_risk"
    elif q >= 35:
        level = "caution"
    else:
        level = "watch"

    search_text_parts = [
        "Q_risk",
        level,
        industry_sub,
        asset_name,
        customer_type,
        main_bank,
        competitor,
        deal_source,
        status,
        *(tags[:6]),
    ]

    return {
        "score": round(q, 1),
        "level": level,
        "patterns": tags[:10],
        "reasons": reasons[:6],
        "similarity_text": " / ".join(part for part in search_text_parts if part),
    }


def _existing_qrisk(data: dict[str, Any]) -> float | None:
    candidates = [
        data.get("q_risk_score"),
        data.get("quantum_risk"),
        _get_nested(data, "result", "q_risk_score"),
        _get_nested(data, "result", "quantum_risk"),
        _get_nested(data, "result", "aurion", "q_risk", "score"),
    ]
    for value in candidates:
        num = _num(value)
        if num is not None and 0 <= num <= 100:
            return round(num, 1)
    return None


def _level_for_score(score: float) -> str:
    if score >= 60:
        return "high_risk"
    if score >= 35:
        return "caution"
    return "watch"


def _similarity_text_with_level(similarity_text: str, level: str) -> str:
    parts = [part.strip() for part in str(similarity_text or "").split("/") if part.strip()]
    if len(parts) >= 2 and parts[0] == "Q_risk":
        parts[1] = level
        return " / ".join(parts)
    if parts and parts[0] == "Q_risk":
        return " / ".join(["Q_risk", level, *parts[1:]])
    return " / ".join(["Q_risk", level, *parts])


def enrich_case_data(data: dict[str, Any], *, row_key: str) -> tuple[dict[str, Any], bool]:
    computed = compute_qrisk(data, row_key=row_key)
    existing = _existing_qrisk(data)
    score = existing if existing is not None else computed["score"]
    level = _level_for_score(score)
    similarity_text = _similarity_text_with_level(computed["similarity_text"], level)
    changed = False

    if data.get("q_risk_score") != score:
        data["q_risk_score"] = score
        changed = True
    if data.get("quantum_risk") != score:
        data["quantum_risk"] = score
        changed = True
    if data.get("q_risk_level") != level:
        data["q_risk_level"] = level
        changed = True
    if data.get("q_risk_role") != "discovery_signal":
        data["q_risk_role"] = "discovery_signal"
        changed = True
    for key, value in (
        ("q_risk_patterns", computed["patterns"]),
        ("q_risk_reasons", computed["reasons"]),
        ("qrisk_similarity_text", similarity_text),
    ):
        if data.get(key) != value:
            data[key] = value
            changed = True

    result = data.setdefault("result", {})
    if isinstance(result, dict):
        for key in ("q_risk_score", "quantum_risk"):
            if result.get(key) != score:
                result[key] = score
                changed = True
        aurion = result.setdefault("aurion", {})
        if isinstance(aurion, dict):
            expected = {
                "score": score,
                "level": level,
                "role": "discovery_signal",
                "patterns": computed["patterns"],
                "reasons": computed["reasons"],
            }
            if aurion.get("q_risk") != expected:
                aurion["q_risk"] = expected
                changed = True

    data["qrisk_backfilled_at"] = "2026-07-13"
    return data, changed


def backfill_past_cases(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT id, data FROM past_cases").fetchall()
    updated = 0
    for case_id, raw in rows:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            continue
        data, changed = enrich_case_data(data, row_key=str(case_id))
        if not changed and "qrisk_backfilled_at" in data:
            continue

        score = _num(_get_nested(data, "result", "score")) or _num(data.get("score"))
        user_eq = _num(_get_nested(data, "result", "user_eq")) or _num(data.get("user_eq"))
        status = _text(data.get("final_status"))
        industry_sub = _text(data.get("industry_sub") or _get_nested(data, "inputs", "industry_sub"))
        conn.execute(
            """
            UPDATE past_cases
               SET data = ?,
                   score = COALESCE(?, score),
                   user_eq = COALESCE(?, user_eq),
                   final_status = COALESCE(NULLIF(?, ''), final_status),
                   industry_sub = COALESCE(NULLIF(?, ''), industry_sub)
             WHERE id = ?
            """,
            (
                json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                score,
                user_eq,
                status,
                industry_sub,
                case_id,
            ),
        )
        updated += 1
    return {"seen": len(rows), "updated": updated}


def backfill_screening_records(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT id, case_id, total_score, asset_score, tenant_score,
               q_risk_score, competitor_pressure_score, outcome, input_snapshot
          FROM screening_records
        """
    ).fetchall()
    updated = 0
    for row in rows:
        (
            row_id,
            case_id,
            total_score,
            asset_score,
            tenant_score,
            existing_qrisk,
            competitor_pressure,
            outcome,
            raw_snapshot,
        ) = row
        try:
            snapshot = json.loads(raw_snapshot or "{}")
        except json.JSONDecodeError:
            snapshot = {}
        payload = dict(snapshot)
        payload.update(
            {
                "score": total_score,
                "total_score": total_score,
                "asset_score": asset_score,
                "tenant_score": tenant_score,
                "outcome": outcome,
                "competitor_pressure_score": competitor_pressure,
            }
        )
        computed = compute_qrisk(payload, row_key=str(case_id or row_id))
        q_score = _num(existing_qrisk)
        if q_score is None or q_score <= 0:
            q_score = computed["score"]
        else:
            q_score = round(q_score, 1)
        level = _level_for_score(q_score)
        similarity_text = _similarity_text_with_level(computed["similarity_text"], level)
        snapshot.update(
            {
                "q_risk_score": q_score,
                "quantum_risk": q_score,
                "q_risk_level": level,
                "q_risk_role": "discovery_signal",
                "q_risk_patterns": computed["patterns"],
                "q_risk_reasons": computed["reasons"],
                "qrisk_similarity_text": similarity_text,
                "qrisk_backfilled_at": "2026-07-13",
            }
        )
        conn.execute(
            """
            UPDATE screening_records
               SET q_risk_score = ?,
                   input_snapshot = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (
                q_score,
                json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
                dt.datetime.now(dt.timezone.utc).isoformat(),
                row_id,
            ),
        )
        updated += 1
    return {"seen": len(rows), "updated": updated}


def summarize(conn: sqlite3.Connection) -> dict[str, Any]:
    past_total = conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0]
    past_q = conn.execute(
        "SELECT COUNT(*) FROM past_cases WHERE data LIKE '%\"q_risk_score\"%'"
    ).fetchone()[0]
    rec_total = conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()[0]
    rec_q = conn.execute(
        "SELECT COUNT(*) FROM screening_records WHERE q_risk_score IS NOT NULL"
    ).fetchone()[0]
    levels: dict[str, int] = {}
    for (level,) in conn.execute(
        """
        SELECT json_extract(data, '$.q_risk_level')
          FROM past_cases
         WHERE json_extract(data, '$.q_risk_level') IS NOT NULL
        """
    ):
        levels[str(level)] = levels.get(str(level), 0) + 1
    return {
        "past_cases": {"total": past_total, "with_qrisk": past_q, "levels": levels},
        "screening_records": {"total": rec_total, "with_qrisk": rec_q},
    }


def run(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("BEGIN")
        past = backfill_past_cases(conn)
        records = backfill_screening_records(conn)
        conn.commit()
        return {"db": str(db_path), "past": past, "screening_records": records, "summary": summarize(conn)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", nargs="?", default=str(DEFAULT_DB))
    args = parser.parse_args()
    result = run(Path(args.db))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
