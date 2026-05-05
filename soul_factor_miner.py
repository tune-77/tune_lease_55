from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from itertools import combinations
import re
from typing import Any, Callable

import numpy as np

from data_cases import DB_PATH

VALID_STATUSES = ("成約", "失注")
SUCCESS_STATUSES = ("成約",)

MISSING_STRINGS = {
    "",
    "0",
    "0.0",
    "0%",
    "None",
    "未設定",
    "未読取",
    "無格付",
}

_CACHE: dict[str, Any] = {"db_mtime": None, "result": None}
_REVERSE_BONUS_CACHE: dict[str, Any] = {"db_mtime": None, "result": None}
_PATCH_CACHE: dict[str, Any] = {"db_mtime": None, "result": None}
_MAHALANOBIS_MODEL: Any = None


def _db_mtime() -> float:
    return os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0.0


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        s = str(val).strip().replace(",", "")
        if s in MISSING_STRINGS:
            return None
        return float(s)
    except Exception:
        return None


def _norm_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s in MISSING_STRINGS:
        return None
    return s


def _grade_norm(val: Any) -> str | None:
    s = _norm_str(val)
    if s is None:
        return None
    if "要注意" in s:
        return "要注意先"
    if "無格付" in s or s == "0":
        return "無格付"
    if s in {"1", "2", "3", "1-3", "①1-3 (優良)", "①"}:
        return "1-3"
    if s in {"4", "5", "6", "4-6", "②4-6 (標準)", "②"}:
        return "4-6"
    if s in {"7", "8", "9", "7-9", "③要注意以下", "③"}:
        return "要注意先"
    return s


def _get_raw(case: dict, *keys: str) -> Any:
    inputs = case.get("inputs") or {}
    for key in keys:
        if key in case and case.get(key) not in (None, ""):
            return case.get(key)
        if key in inputs and inputs.get(key) not in (None, ""):
            return inputs.get(key)
    return None


def _get_float(case: dict, *keys: str) -> float | None:
    return _safe_float(_get_raw(case, *keys))


def _op_margin(case: dict) -> float | None:
    sales = _get_float(case, "nenshu")
    op_profit = _get_float(case, "op_profit", "rieki")
    if sales in (None, 0) or op_profit is None:
        return None
    return op_profit / sales


def _gross_margin(case: dict) -> float | None:
    sales = _get_float(case, "nenshu")
    gross_profit = _get_float(case, "gross_profit")
    if sales in (None, 0) or gross_profit is None:
        return None
    return gross_profit / sales


def _bank_sales_ratio(case: dict) -> float | None:
    sales = _get_float(case, "nenshu")
    bank_credit = _get_float(case, "bank_credit")
    if sales in (None, 0) or bank_credit is None:
        return None
    return bank_credit / sales


def _lease_sales_ratio(case: dict) -> float | None:
    sales = _get_float(case, "nenshu")
    lease_credit = _get_float(case, "lease_credit")
    if sales in (None, 0) or lease_credit is None:
        return None
    return lease_credit / sales


def _debt_sales_ratio(case: dict) -> float | None:
    sales = _get_float(case, "nenshu")
    bank_credit = _get_float(case, "bank_credit")
    lease_credit = _get_float(case, "lease_credit")
    if sales in (None, 0) or bank_credit is None or lease_credit is None:
        return None
    return (bank_credit + lease_credit) / sales


def _dep_profit_ratio(case: dict) -> float | None:
    op_profit = _get_float(case, "op_profit", "rieki")
    depreciation = _get_float(case, "depreciation", "dep_expense")
    if op_profit in (None, 0) or depreciation is None:
        return None
    return depreciation / abs(op_profit)


def _equity_ratio(case: dict) -> float | None:
    net_assets = _get_float(case, "net_assets")
    total_assets = _get_float(case, "total_assets")
    if net_assets is None or total_assets in (None, 0):
        return None
    return net_assets / total_assets


def _q_risk(case: dict) -> float | None:
    result = case.get("result") or {}
    for key in ("quantum_risk", "quantum_explained_risk"):
        val = result.get(key)
        if val is None:
            continue
        try:
            val_f = float(val)
        except Exception:
            continue
        if math.isfinite(val_f):
            return val_f
    return None


def _load_mahalanobis_model():
    global _MAHALANOBIS_MODEL
    if _MAHALANOBIS_MODEL is not None:
        return _MAHALANOBIS_MODEL
    model_path = os.path.join(os.path.dirname(DB_PATH), "mahalanobis_model.joblib")
    if not os.path.exists(model_path):
        _MAHALANOBIS_MODEL = None
        return None
    try:
        from mahalanobis_engine import MahalanobisScorer
        _MAHALANOBIS_MODEL = MahalanobisScorer.load(model_path)
    except Exception:
        _MAHALANOBIS_MODEL = None
    return _MAHALANOBIS_MODEL


def _mahalanobis_distance(case: dict) -> float | None:
    model = _load_mahalanobis_model()
    if model is None:
        return None
    try:
        from train_mahalanobis import FEATURES, _extract_val
    except Exception:
        return None
    try:
        row = [float(_extract_val(case, f) or 0.0) for f in FEATURES]
        _, d, _, _ = model.get_analysis(row)
        if math.isfinite(float(d)):
            return float(d)
    except Exception:
        return None
    return None


def _build_numeric_threshold_atoms(
    field: str,
    getter: Callable[[dict], float | None],
    cases: list[dict],
    top_k: int = 4,
) -> list[tuple[str, Callable[[dict], bool]]]:
    rows: list[tuple[float, int]] = []
    for case in cases:
        value = getter(case)
        if value is None or not math.isfinite(value):
            continue
        rows.append((float(value), 1 if case.get("final_status") in SUCCESS_STATUSES else 0))
    if len(rows) < 20:
        return []

    arr = np.array(rows, dtype=float)
    vals = arr[:, 0]
    labels = arr[:, 1].astype(int)
    order = np.argsort(vals, kind="mergesort")
    vals = vals[order]
    labels = labels[order]

    unique_vals = np.unique(vals)
    if unique_vals.size < 5:
        return []

    total_success = int(labels.sum())
    total_loss = int(len(labels) - total_success)
    if total_success == 0 or total_loss == 0:
        return []

    cum_success = np.cumsum(labels)
    cum_loss = np.cumsum(1 - labels)

    candidates: list[dict[str, Any]] = []
    for t in unique_vals:
        left = int(np.searchsorted(vals, t, side="left"))
        right = int(np.searchsorted(vals, t, side="right"))

        ge_success = total_success - (int(cum_success[left - 1]) if left > 0 else 0)
        ge_loss = total_loss - (int(cum_loss[left - 1]) if left > 0 else 0)
        le_success = int(cum_success[right - 1]) if right > 0 else 0
        le_loss = int(cum_loss[right - 1]) if right > 0 else 0

        for direction, success_count, loss_count in (
            (">=", ge_success, ge_loss),
            ("<=", le_success, le_loss),
        ):
            if success_count < max(20, int(total_success * 0.08)):
                continue
            coverage = success_count / total_success if total_success else 0.0
            loss_cov = loss_count / total_loss if total_loss else 0.0
            precision = success_count / (success_count + loss_count) if (success_count + loss_count) else 0.0
            score = coverage - (0.85 * loss_cov) + (0.05 * precision)
            candidates.append({
                "score": score,
                "coverage": coverage,
                "loss_cov": loss_cov,
                "precision": precision,
                "t": float(t),
                "direction": direction,
                "success_count": int(success_count),
                "loss_count": int(loss_count),
            })

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["coverage"],
            x["precision"],
            x["success_count"],
        ),
        reverse=True,
    )

    atoms: list[tuple[str, Callable[[dict], bool]]] = []
    for cand in candidates[:top_k]:
        t = cand["t"]
        direction = cand["direction"]
        label = f"{field} {direction} {t:.4g}"

        def pred(case: dict, getter=getter, t=t, direction=direction) -> bool:
            value = getter(case)
            if value is None or not math.isfinite(value):
                return False
            return value >= t if direction == ">=" else value <= t

        atoms.append((label, pred))
    return atoms


def _build_cases() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM past_cases WHERE final_status IN ('成約', '失注') ORDER BY timestamp ASC"
        )
        cases: list[dict] = []
        for (raw,) in cursor.fetchall():
            try:
                case = json.loads(raw)
            except Exception:
                continue
            if case.get("final_status") not in VALID_STATUSES:
                continue
            cases.append(case)
        return cases


def _predict_hantei_score(case: dict) -> float | None:
    try:
        result = case.get("result") or {}
        qual_corr = result.get("qualitative_scoring_correction") or {}
        score = result.get("score")
        if isinstance(qual_corr, dict):
            combined = qual_corr.get("combined_score")
            if combined is not None:
                return float(combined)
        if score is not None:
            return float(score)
    except Exception:
        return None
    return None


def _build_atoms(cases: list[dict]) -> list[tuple[str, Callable[[dict], bool]]]:
    atoms: list[tuple[str, Callable[[dict], bool]]] = []

    # 直接入力のカテゴリ値
    cat_specs: list[tuple[str, Callable[[dict], Any], Callable[[Any], Any] | None, int]] = [
        ("sales_dept", lambda c: _get_raw(c, "sales_dept"), None, 10),
        ("industry_major", lambda c: _get_raw(c, "industry_major"), None, 20),
        ("industry_sub", lambda c: _get_raw(c, "industry_sub"), None, 20),
        ("grade", lambda c: _get_raw(c, "grade"), _grade_norm, 20),
        ("customer_type", lambda c: _get_raw(c, "customer_type"), None, 20),
        ("deal_source", lambda c: _get_raw(c, "deal_source"), None, 20),
        ("competitor", lambda c: _get_raw(c, "competitor"), None, 20),
        ("contract_type", lambda c: _get_raw(c, "contract_type"), None, 20),
    ]
    for field, getter, norm, min_count in cat_specs:
        counter: dict[str, int] = {}
        for case in cases:
            value = getter(case)
            value = norm(value) if norm else _norm_str(value)
            if value is None:
                continue
            counter[value] = counter.get(value, 0) + 1
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if field == "industry_sub":
            items = items[:8]
        elif field == "contract_type":
            items = items[:6]
        elif field == "sales_dept":
            items = items[:5]
        elif field == "grade":
            items = items[:10]
        for value, count in items:
            if count < min_count:
                continue

            def pred(case: dict, getter=getter, value=value, norm=norm) -> bool:
                current = getter(case)
                current = norm(current) if norm else _norm_str(current)
                return current == value

            atoms.append((f"{field}={value}", pred))

    # 数値系の量的条件。入力項目由来だけに限定。
    numeric_specs: list[tuple[str, Callable[[dict], float | None]]] = [
        ("nenshu", lambda c: _get_float(c, "nenshu")),
        ("gross_profit", lambda c: _get_float(c, "gross_profit")),
        ("op_profit", lambda c: _get_float(c, "op_profit", "rieki")),
        ("ord_profit", lambda c: _get_float(c, "ord_profit")),
        ("net_income", lambda c: _get_float(c, "net_income")),
        ("machines", lambda c: _get_float(c, "machines")),
        ("other_assets", lambda c: _get_float(c, "other_assets")),
        ("total_assets", lambda c: _get_float(c, "total_assets")),
        ("net_assets", lambda c: _get_float(c, "net_assets")),
        ("rent", lambda c: _get_float(c, "rent")),
        ("depreciation", lambda c: _get_float(c, "depreciation", "dep_expense")),
        ("rent_expense", lambda c: _get_float(c, "rent_expense")),
        ("bank_credit", lambda c: _get_float(c, "bank_credit")),
        ("lease_credit", lambda c: _get_float(c, "lease_credit")),
        ("contracts", lambda c: _get_float(c, "contracts")),
        ("lease_term", lambda c: _get_float(c, "lease_term")),
        ("acquisition_cost", lambda c: _get_float(c, "acquisition_cost")),
        ("lease_asset_score", lambda c: _get_float(c, "lease_asset_score")),
        ("op_margin", _op_margin),
        ("gross_margin", _gross_margin),
        ("bank_sales_ratio", _bank_sales_ratio),
        ("lease_sales_ratio", _lease_sales_ratio),
        ("debt_sales_ratio", _debt_sales_ratio),
        ("dep_profit_ratio", _dep_profit_ratio),
        ("equity_ratio", _equity_ratio),
        ("q_risk", _q_risk),
        ("mahalanobis_distance", _mahalanobis_distance),
        ("profit_to_assets", lambda c: (_get_float(c, "op_profit", "rieki") / _get_float(c, "total_assets")) if (_get_float(c, "op_profit", "rieki") not in (None, 0) and _get_float(c, "total_assets") not in (None, 0)) else None),
        ("sales_to_assets", lambda c: (_get_float(c, "nenshu") / _get_float(c, "total_assets")) if (_get_float(c, "nenshu") not in (None, 0) and _get_float(c, "total_assets") not in (None, 0)) else None),
        ("lease_to_assets", lambda c: (_get_float(c, "lease_credit") / _get_float(c, "total_assets")) if (_get_float(c, "lease_credit") not in (None, 0) and _get_float(c, "total_assets") not in (None, 0)) else None),
    ]
    for field, getter in numeric_specs:
        atoms.extend(_build_numeric_threshold_atoms(field, getter, cases))

    # 重複ラベルを消す
    unique: list[tuple[str, Callable[[dict], bool]]] = []
    seen: set[str] = set()
    for label, pred in atoms:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, pred))
    return unique


def _build_bitsets(cases: list[dict], atoms: list[tuple[str, Callable[[dict], bool]]]) -> tuple[list[int], list[int]]:
    success_bits: list[int] = []
    loss_bits: list[int] = []
    for label, pred in atoms:
        sb = 0
        lb = 0
        for idx, case in enumerate(cases):
            try:
                if not pred(case):
                    continue
            except Exception:
                continue
            if case.get("final_status") in SUCCESS_STATUSES:
                sb |= 1 << idx
            else:
                lb |= 1 << idx
        success_bits.append(sb)
        loss_bits.append(lb)
    return success_bits, loss_bits


def _atom_source_keys(label: str) -> set[str]:
    keys: set[str] = set()
    pieces = re.findall(r"\[([^\[\]]+)\]", label)
    if not pieces:
        pieces = [label]
    for piece in pieces:
        chunk = piece.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key = chunk.split("=", 1)[0].strip()
        elif " " in chunk:
            key = chunk.split(" ", 1)[0].strip()
        else:
            key = chunk.strip()
        if key.startswith("VAR"):
            continue
        keys.add(key)
    return keys


def _make_derived_pair_variables(
    cases: list[dict],
    atoms: list[tuple[str, Callable[[dict], bool]]],
    success_bits: list[int],
    loss_bits: list[int],
    n_success: int,
    n_loss: int,
    top_k: int = 18,
) -> list[tuple[str, Callable[[dict], bool]]]:
    candidates: list[dict[str, Any]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            sb = success_bits[i] & success_bits[j]
            lb = loss_bits[i] & loss_bits[j]
            sc = sb.bit_count()
            lc = lb.bit_count()
            if sc < max(40, int(n_success * 0.12)):
                continue
            if sc == 0:
                continue
            success_coverage = sc / n_success if n_success else 0.0
            loss_coverage = lc / n_loss if n_loss else 0.0
            precision = sc / (sc + lc) if (sc + lc) else 0.0
            score = success_coverage - (0.8 * loss_coverage)
            candidates.append({
                "score": score,
                "success_coverage": success_coverage,
                "loss_coverage": loss_coverage,
                "precision": precision,
                "success_count": sc,
                "loss_count": lc,
                "i": i,
                "j": j,
            })

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["success_coverage"],
            x["precision"],
            x["success_count"],
        ),
        reverse=True,
    )

    selected: list[tuple[str, Callable[[dict], bool]]] = []
    seen: set[str] = set()
    for cand in candidates[: max(top_k * 3, top_k)]:
        i = cand["i"]
        j = cand["j"]
        label = f"VAR[{atoms[i][0]}] & [{atoms[j][0]}]"
        if label in seen:
            continue
        seen.add(label)

        def pred(case: dict, pa=atoms[i][1], pb=atoms[j][1]) -> bool:
            return bool(pa(case)) and bool(pb(case))

        selected.append((label, pred))
        if len(selected) >= top_k:
            break
    return selected


def _make_derived_triplet_variables(
    cases: list[dict],
    atoms: list[tuple[str, Callable[[dict], bool]]],
    success_bits: list[int],
    loss_bits: list[int],
    n_success: int,
    n_loss: int,
    top_k: int = 12,
) -> list[tuple[str, Callable[[dict], bool]]]:
    candidates: list[dict[str, Any]] = []
    for i, j, k in combinations(range(len(atoms)), 3):
        sb = success_bits[i] & success_bits[j] & success_bits[k]
        lb = loss_bits[i] & loss_bits[j] & loss_bits[k]
        sc = sb.bit_count()
        lc = lb.bit_count()
        if sc < max(25, int(n_success * 0.08)):
            continue
        if sc == 0:
            continue
        success_coverage = sc / n_success if n_success else 0.0
        loss_coverage = lc / n_loss if n_loss else 0.0
        precision = sc / (sc + lc) if (sc + lc) else 0.0
        score = success_coverage - (0.95 * loss_coverage) + (0.05 * precision)
        candidates.append({
            "score": score,
            "success_coverage": success_coverage,
            "loss_coverage": loss_coverage,
            "precision": precision,
            "success_count": sc,
            "loss_count": lc,
            "i": i,
            "j": j,
            "k": k,
        })

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["success_coverage"],
            x["precision"],
            x["success_count"],
        ),
        reverse=True,
    )

    selected: list[tuple[str, Callable[[dict], bool]]] = []
    seen: set[str] = set()
    for cand in candidates[: max(top_k * 4, top_k)]:
        i = cand["i"]
        j = cand["j"]
        k = cand["k"]
        label = f"VAR3[{atoms[i][0]}] & [{atoms[j][0]}] & [{atoms[k][0]}]"
        if label in seen:
            continue
        seen.add(label)

        def pred(case: dict, pa=atoms[i][1], pb=atoms[j][1], pc=atoms[k][1]) -> bool:
            return bool(pa(case)) and bool(pb(case)) and bool(pc(case))

        selected.append((label, pred))
        if len(selected) >= top_k:
            break
    return selected


def _pattern_metrics(
    idxs: tuple[int, ...],
    atoms: list[tuple[str, Callable[[dict], bool]]],
    success_bits: list[int],
    loss_bits: list[int],
    n_success: int,
    n_loss: int,
) -> dict[str, Any]:
    source_keys: set[str] = set()
    for idx in idxs:
        source_keys |= _atom_source_keys(atoms[idx][0])
    if len(source_keys) < len(idxs):
        return {
            "pattern": " & ".join(atoms[i][0] for i in idxs),
            "items": [atoms[i][0] for i in idxs],
            "n_items": len(idxs),
            "success_count": 0,
            "loss_count": 0,
            "support": 0,
            "success_coverage": 0.0,
            "loss_coverage": 1.0,
            "precision": 0.0,
            "lift": 0.0,
            "invalid": True,
        }
    sb = (1 << 0) - 1
    lb = (1 << 0) - 1
    sb = success_bits[idxs[0]]
    lb = loss_bits[idxs[0]]
    for idx in idxs[1:]:
        sb &= success_bits[idx]
        lb &= loss_bits[idx]
    success_count = sb.bit_count()
    loss_count = lb.bit_count()
    support = success_count + loss_count
    success_coverage = success_count / n_success if n_success else 0.0
    loss_coverage = loss_count / n_loss if n_loss else 0.0
    precision = success_count / support if support else 0.0
    lift = (success_coverage / (support / (n_success + n_loss))) if support else 0.0
    support_diff = success_coverage - loss_coverage
    return {
        "pattern": " & ".join(atoms[i][0] for i in idxs),
        "items": [atoms[i][0] for i in idxs],
        "n_items": len(idxs),
        "success_count": success_count,
        "loss_count": loss_count,
        "support": support,
        "success_coverage": success_coverage,
        "loss_coverage": loss_coverage,
        "support_diff": support_diff,
        "precision": precision,
        "lift": lift,
        "invalid": False,
    }


def mine_soul_factors(force_recompute: bool = False) -> dict[str, Any] | None:
    db_mtime = _db_mtime()
    if not force_recompute and _CACHE.get("result") is not None and _CACHE.get("db_mtime") == db_mtime:
        return _CACHE["result"]

    cases = _build_cases()
    if not cases:
        return None

    n_total = len(cases)
    n_success = sum(1 for c in cases if c.get("final_status") in SUCCESS_STATUSES)
    n_loss = n_total - n_success
    threshold = 71.0
    fn_cases = []
    tp_cases = []
    fp_cases = []
    tn_cases = []
    for c in cases:
        score = _predict_hantei_score(c)
        if score is None:
            continue
        pred_success = score >= threshold
        actual_success = c.get("final_status") in SUCCESS_STATUSES
        if actual_success and pred_success:
            tp_cases.append(c)
        elif actual_success and not pred_success:
            fn_cases.append(c)
        elif (not actual_success) and pred_success:
            fp_cases.append(c)
        else:
            tn_cases.append(c)
    base_atoms = _build_atoms(cases)
    atoms = list(base_atoms)
    success_bits, loss_bits = _build_bitsets(cases, atoms)

    derived_atoms = _make_derived_pair_variables(cases, base_atoms, success_bits[: len(base_atoms)], loss_bits[: len(base_atoms)], n_success, n_loss)
    if derived_atoms:
        atoms = atoms + derived_atoms
        success_bits, loss_bits = _build_bitsets(cases, atoms)

    derived_triplets = _make_derived_triplet_variables(cases, base_atoms, success_bits[: len(base_atoms)], loss_bits[: len(base_atoms)], n_success, n_loss)
    if derived_triplets:
        atoms = atoms + derived_triplets
        success_bits, loss_bits = _build_bitsets(cases, atoms)

    exact_patterns: list[dict[str, Any]] = []
    best_patterns: list[dict[str, Any]] = []
    best_atoms: list[dict[str, Any]] = []

    for i in range(len(atoms)):
        m = _pattern_metrics((i,), atoms, success_bits, loss_bits, n_success, n_loss)
        best_atoms.append(m)

    best_atoms.sort(key=lambda x: (x["success_coverage"] - x["loss_coverage"], x["success_coverage"], x["precision"], x["support"]), reverse=True)

    for i, j in combinations(range(len(atoms)), 2):
        m = _pattern_metrics((i, j), atoms, success_bits, loss_bits, n_success, n_loss)
        if m.get("invalid"):
            continue
        if m["success_coverage"] >= 0.8 and m["loss_count"] == 0:
            exact_patterns.append(m)
        if m["success_count"] >= 50:
            best_patterns.append(m)
    for i, j, k in combinations(range(len(atoms)), 3):
        m = _pattern_metrics((i, j, k), atoms, success_bits, loss_bits, n_success, n_loss)
        if m.get("invalid"):
            continue
        if m["success_coverage"] >= 0.8 and m["loss_count"] == 0:
            exact_patterns.append(m)
        if m["success_count"] >= 50:
            best_patterns.append(m)

    exact_patterns.sort(key=lambda x: (x["success_coverage"], x["success_count"], -x["n_items"]), reverse=True)
    best_patterns.sort(key=lambda x: (x["support_diff"], x["success_coverage"], x["precision"], x["support"]), reverse=True)

    result = {
        "n_cases": n_total,
        "n_success": n_success,
        "n_loss": n_loss,
        "n_fn": len(fn_cases),
        "n_fp": len(fp_cases),
        "n_tp": len(tp_cases),
        "n_tn": len(tn_cases),
        "n_atoms": len(atoms),
        "search_space": {
            "pairs": math.comb(len(atoms), 2) if len(atoms) >= 2 else 0,
            "triplets": math.comb(len(atoms), 3) if len(atoms) >= 3 else 0,
            "total": (math.comb(len(atoms), 2) + math.comb(len(atoms), 3)) if len(atoms) >= 3 else math.comb(len(atoms), 2),
        },
        "exact_patterns": exact_patterns[:50],
        "best_patterns": best_patterns[:50],
        "single_patterns": best_atoms[:50],
        "best_atoms": best_atoms[:30],
        "derived_variables": [label for label, _ in derived_atoms[:30]],
        "derived_triplet_variables": [label for label, _ in derived_triplets[:20]],
        "fn_patterns": [],
        "criteria": {
            "success_coverage_min": 0.8,
            "loss_count_max": 0,
            "source": "yesterday_direct_input_fields_only_plus_reference_vars",
        },
    }

    # FNにだけ多い項目を別集計
    fn_atoms = []
    if fn_cases:
        fn_atoms = _build_atoms(fn_cases)
        fn_bits, _ = _build_bitsets(fn_cases, fn_atoms)
        # 成約全体との比較ではなく、FN対成功全体の出現差を見る
        fn_rows = []
        success_rows = []
        for label, pred in fn_atoms:
            fn_c = sum(1 for c in fn_cases if pred(c))
            su_c = sum(1 for c in cases if c.get("final_status") in SUCCESS_STATUSES and pred(c))
            fn_rate = fn_c / len(fn_cases) if fn_cases else 0.0
            su_rate = su_c / n_success if n_success else 0.0
            fn_rows.append({
                "pattern": label,
                "fn_count": fn_c,
                "success_count": su_c,
                "fn_rate": fn_rate,
                "success_rate": su_rate,
                "diff": fn_rate - su_rate,
            })
            success_rows.append({
                "pattern": label,
                "success_count": su_c,
                "fn_count": fn_c,
                "success_rate": su_rate,
                "fn_rate": fn_rate,
                "diff": su_rate - fn_rate,
            })
        fn_rows.sort(key=lambda x: (x["diff"], x["fn_rate"], x["success_rate"]), reverse=True)
        success_rows.sort(key=lambda x: (x["diff"], x["success_rate"], x["fn_rate"]), reverse=True)
        result["fn_only_patterns"] = fn_rows[:50]
        result["success_overall_patterns"] = success_rows[:50]
    else:
        result["fn_only_patterns"] = []
        result["success_overall_patterns"] = []

    _CACHE.update({"db_mtime": db_mtime, "result": result})
    return result


def _reverse_bonus_gate(case: dict) -> bool:
    return (
        _norm_str(_get_raw(case, "deal_source")) == "その他"
        and _norm_str(_get_raw(case, "customer_type")) == "新規先"
    )


def _reverse_bonus_grade(case: dict) -> str | None:
    return _grade_norm(_get_raw(case, "grade"))


def _reverse_bonus_sales(case: dict) -> float | None:
    return _get_float(case, "nenshu")


def _reverse_bonus_bank_credit(case: dict) -> float | None:
    return _get_float(case, "bank_credit")


def _reverse_bonus_bank_ratio(case: dict) -> float | None:
    sales = _reverse_bonus_sales(case)
    bank = _reverse_bonus_bank_credit(case)
    if sales in (None, 0) or bank is None:
        return None
    return bank / sales


def _reverse_bonus_atom_hit(case: dict, atom: dict[str, Any]) -> bool:
    if atom["kind"] == "categorical":
        current = _reverse_bonus_grade(case)
        return current == atom["threshold"]
    if atom["key"] == "sales":
        value = _reverse_bonus_sales(case)
    elif atom["key"] == "bank_credit":
        value = _reverse_bonus_bank_credit(case)
    else:
        value = _reverse_bonus_bank_ratio(case)
    if value is None or not math.isfinite(float(value)):
        return False
    if atom["op"] == ">=":
        return float(value) >= float(atom["threshold"])
    return float(value) <= float(atom["threshold"])


def _reverse_bonus_atom_text(atom: dict[str, Any]) -> str:
    if atom["kind"] == "categorical":
        return f"{atom['key']}={atom['threshold']}"
    return f"{atom['key']} {atom['op']} {atom['threshold']:.6g}"


def _reverse_bonus_atom_jp(atom: dict[str, Any]) -> str:
    label_map = {
        "sales": "売上高",
        "bank_credit": "銀行借入",
        "bank_ratio": "売上に対する銀行借入比率",
        "grade": "格付",
    }
    key_label = label_map.get(atom["key"], atom["key"])
    if atom["kind"] == "categorical":
        if atom["threshold"] is None:
            return f"{key_label} が空欄"
        return f"{key_label} が {atom['threshold']}"
    if atom["op"] == ">=":
        return f"{key_label} が {atom['threshold']:.6g} 以上"
    return f"{key_label} が {atom['threshold']:.6g} 以下"


def _reverse_bonus_rule_conditions(rule: dict[str, Any], activation_threshold: float) -> list[str]:
    conditions = [
        "deal_source=その他",
        "customer_type=新規先",
        f"posterior>={activation_threshold:.2f}",
        "other_count=0",
        f"fn_count>={max(3, int(rule.get('fn_count') or 0))}",
    ]
    predicates = rule.get("predicates") or []
    conditions.extend(_reverse_bonus_atom_text(atom) for atom in predicates)
    return conditions


def _reverse_bonus_rule_description(rule: dict[str, Any], activation_threshold: float) -> str:
    predicates = rule.get("predicates") or []
    jp_predicates = "、".join(_reverse_bonus_atom_jp(atom) for atom in predicates)
    return (
        "以下の条件を同時に満たしたときだけ逆転のベイズ加点を発動する。"
        f" 対象は deal_source=その他 かつ customer_type=新規先 に限定し、"
        f" 後方確率が {activation_threshold:.2f} 以上、かつ FN の偏りが強い組み合わせのみを採用する。"
        f" 個別条件は {jp_predicates}。"
    )


def _reverse_bonus_predict_score(case: dict) -> float | None:
    return _predict_hantei_score(case)


def _reverse_bonus_consistent(bounds: dict[str, tuple[str, float]], key: str, op: str, value: float) -> bool:
    prev = bounds.get(key)
    if prev is None:
        return True
    prev_op, prev_value = prev
    if prev_op == op:
        if op == ">=":
            return value >= prev_value
        return value <= prev_value
    if prev_op == ">=" and op == "<=":
        return prev_value <= value
    if prev_op == "<=" and op == ">=":
        return value <= prev_value
    return True


def mine_reverse_bayes_bonus(force_recompute: bool = False) -> dict[str, Any] | None:
    db_mtime = _db_mtime()
    if not force_recompute and _REVERSE_BONUS_CACHE.get("result") is not None and _REVERSE_BONUS_CACHE.get("db_mtime") == db_mtime:
        return _REVERSE_BONUS_CACHE["result"]

    cases = _build_cases()
    if not cases:
        return None

    gate_cases = [c for c in cases if _reverse_bonus_gate(c)]
    if len(gate_cases) < 6:
        return None

    threshold = 71.0
    fn_cases: list[dict] = []
    other_cases: list[dict] = []
    for case in gate_cases:
        score = _reverse_bonus_predict_score(case)
        actual_success = case.get("final_status") in SUCCESS_STATUSES
        is_fn = actual_success and score is not None and score < threshold
        if is_fn:
            fn_cases.append(case)
        else:
            other_cases.append(case)

    if not fn_cases:
        return None

    n_fn = len(fn_cases)
    n_other = len(other_cases)

    def _thresholds(values: list[float]) -> list[float]:
        cleaned = sorted({float(v) for v in values if v is not None and math.isfinite(float(v))})
        if len(cleaned) <= 8:
            return cleaned
        idxs = [0, 1, 2, 3, 4, 5, 6, 7]
        out = []
        for i in idxs:
            pos = int(round(i * (len(cleaned) - 1) / (len(idxs) - 1)))
            out.append(cleaned[pos])
        return sorted(set(out))

    sales_values = _thresholds([_reverse_bonus_sales(c) for c in gate_cases if _reverse_bonus_sales(c) is not None])
    bank_values = _thresholds([_reverse_bonus_bank_credit(c) for c in gate_cases if _reverse_bonus_bank_credit(c) is not None])
    ratio_values = _thresholds([_reverse_bonus_bank_ratio(c) for c in gate_cases if _reverse_bonus_bank_ratio(c) is not None])

    grade_values: list[str | None] = []
    seen_grades: set[str | None] = set()
    for case in gate_cases:
        g = _reverse_bonus_grade(case)
        if g in seen_grades:
            continue
        seen_grades.add(g)
        grade_values.append(g)

    atoms: list[dict[str, Any]] = []
    for t in sales_values:
        atoms.append({"label": f"sales <= {t:.4g}", "key": "sales", "op": "<=", "threshold": float(t), "kind": "numeric"})
        atoms.append({"label": f"sales >= {t:.4g}", "key": "sales", "op": ">=", "threshold": float(t), "kind": "numeric"})
    for t in bank_values:
        atoms.append({"label": f"bank_credit <= {t:.4g}", "key": "bank_credit", "op": "<=", "threshold": float(t), "kind": "numeric"})
        atoms.append({"label": f"bank_credit >= {t:.4g}", "key": "bank_credit", "op": ">=", "threshold": float(t), "kind": "numeric"})
    for t in ratio_values:
        atoms.append({"label": f"bank_ratio <= {t:.4g}", "key": "bank_ratio", "op": "<=", "threshold": float(t), "kind": "numeric"})
        atoms.append({"label": f"bank_ratio >= {t:.4g}", "key": "bank_ratio", "op": ">=", "threshold": float(t), "kind": "numeric"})
    for g in grade_values:
        label = "grade=None" if g is None else f"grade={g}"
        atoms.append({"label": label, "key": "grade", "op": "==", "threshold": g, "kind": "categorical"})

    candidates: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for combo in combinations(range(len(atoms)), size):
            selected = [atoms[i] for i in combo]
            keys = [a["key"] for a in selected]
            if len(set(keys)) < 2:
                continue
            if "sales" not in keys:
                continue
            if "bank_ratio" not in keys and "bank_credit" not in keys:
                continue
            bounds: dict[str, tuple[str, float]] = {}
            consistent = True
            for atom in selected:
                key = atom["key"]
                if atom["kind"] == "categorical":
                    continue
                if not _reverse_bonus_consistent(bounds, key, atom["op"], atom["threshold"]):
                    consistent = False
                    break
                prev = bounds.get(key)
                if prev is None:
                    bounds[key] = (atom["op"], atom["threshold"])
                else:
                    prev_op, prev_value = prev
                    if prev_op == atom["op"]:
                        if atom["op"] == ">=":
                            bounds[key] = (atom["op"], max(prev_value, atom["threshold"]))
                        else:
                            bounds[key] = (atom["op"], min(prev_value, atom["threshold"]))
                    elif prev_op == ">=" and atom["op"] == "<=":
                        bounds[key] = (">=", prev_value)
                    elif prev_op == "<=" and atom["op"] == ">=":
                        bounds[key] = (">=", atom["threshold"])
            if not consistent:
                continue

            fn_count = sum(1 for case in fn_cases if all(_reverse_bonus_atom_hit(case, atom) for atom in selected))
            other_count = sum(1 for case in other_cases if all(_reverse_bonus_atom_hit(case, atom) for atom in selected))
            if fn_count == 0:
                continue

            fn_rate = fn_count / n_fn if n_fn else 0.0
            other_rate = other_count / n_other if n_other else 0.0
            prior = n_fn / (n_fn + n_other) if (n_fn + n_other) else 0.5
            fn_lik = (fn_count + 1.0) / (n_fn + 2.0)
            other_lik = (other_count + 1.0) / (n_other + 2.0)
            log_odds = math.log(prior / (1.0 - prior)) + math.log(fn_lik / other_lik)
            posterior = 1.0 / (1.0 + math.exp(-log_odds))
            support_diff = fn_rate - other_rate
            candidates.append({
                "pattern": " & ".join(atom["label"] for atom in selected),
                "items": [atom["label"] for atom in selected],
                "n_items": len(selected),
                "fn_count": fn_count,
                "other_count": other_count,
                "fn_rate": fn_rate,
                "other_rate": other_rate,
                "support_diff": support_diff,
                "posterior": posterior,
                "bayes_factor": fn_lik / other_lik,
                "prior": prior,
                "predicates": selected,
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["other_count"] == 0,
            x["fn_count"],
            x["support_diff"],
            x["posterior"],
            -x["n_items"],
        ),
        reverse=True,
    )

    preferred: list[dict[str, Any]] = []
    manual_specs = [
        {
            "pattern": "sales <= 11972.858 & bank_ratio <= 0.0317367",
            "predicates": [
                {"label": "sales <= 11972.858", "key": "sales", "op": "<=", "threshold": 11972.858, "kind": "numeric"},
                {"label": "bank_ratio <= 0.0317367", "key": "bank_ratio", "op": "<=", "threshold": 0.031736699792146536, "kind": "numeric"},
            ],
        },
        {
            "pattern": "sales >= 100000 & bank_ratio <= 0.22 & grade=無格付",
            "predicates": [
                {"label": "sales >= 100000", "key": "sales", "op": ">=", "threshold": 100000.0, "kind": "numeric"},
                {"label": "bank_ratio <= 0.22", "key": "bank_ratio", "op": "<=", "threshold": 0.22, "kind": "numeric"},
                {"label": "grade=無格付", "key": "grade", "op": "==", "threshold": "無格付", "kind": "categorical"},
            ],
        },
    ]
    for spec in manual_specs:
        fn_count = sum(1 for case in fn_cases if all(_reverse_bonus_atom_hit(case, atom) for atom in spec["predicates"]))
        other_count = sum(1 for case in other_cases if all(_reverse_bonus_atom_hit(case, atom) for atom in spec["predicates"]))
        if fn_count == 0 or other_count != 0:
            continue
        fn_rate = fn_count / n_fn if n_fn else 0.0
        other_rate = other_count / n_other if n_other else 0.0
        prior = n_fn / (n_fn + n_other) if (n_fn + n_other) else 0.5
        fn_lik = (fn_count + 1.0) / (n_fn + 2.0)
        other_lik = (other_count + 1.0) / (n_other + 2.0)
        log_odds = math.log(prior / (1.0 - prior)) + math.log(fn_lik / other_lik)
        posterior = 1.0 / (1.0 + math.exp(-log_odds))
        preferred.append({
            "pattern": spec["pattern"],
            "items": [atom["label"] for atom in spec["predicates"]],
            "n_items": len(spec["predicates"]),
            "fn_count": fn_count,
            "other_count": other_count,
            "fn_rate": fn_rate,
            "other_rate": other_rate,
            "support_diff": fn_rate - other_rate,
            "posterior": posterior,
            "bayes_factor": fn_lik / other_lik,
            "prior": prior,
            "predicates": spec["predicates"],
        })

    small_route = [
        cand for cand in candidates
        if cand["other_count"] == 0
        and any(atom["key"] == "sales" and atom["op"] == "<=" for atom in cand["predicates"])
        and any(atom["key"] == "bank_ratio" and atom["op"] == "<=" for atom in cand["predicates"])
        and not any(atom["key"] == "grade" for atom in cand["predicates"])
    ]
    if small_route:
        preferred.append(max(
            small_route,
            key=lambda x: (
                x["fn_count"],
                x["support_diff"],
                x["posterior"],
                -x["n_items"],
            ),
        ))

    large_route = [
        cand for cand in candidates
        if cand["other_count"] == 0
        and any(atom["key"] == "sales" and atom["op"] == ">=" for atom in cand["predicates"])
        and any(atom["key"] == "bank_ratio" and atom["op"] == "<=" for atom in cand["predicates"])
        and any(
            atom["key"] == "grade" and atom["threshold"] in (None, "無格付")
            for atom in cand["predicates"]
        )
    ]
    if large_route:
        preferred.append(max(
            large_route,
            key=lambda x: (
                x["fn_count"],
                x["support_diff"],
                x["posterior"],
                -x["n_items"],
            ),
        ))

    rules: list[dict[str, Any]] = []
    for cand in preferred + candidates:
        if cand["other_count"] != 0:
            continue
        if any(existing.get("pattern") == cand["pattern"] for existing in rules):
            continue
        bonus_points = int(round(max(3.0, min(12.0, (cand["posterior"] - 0.50) * 20.0))))
        rule = {
            "pattern": cand["pattern"],
            "items": cand["items"],
            "n_items": cand["n_items"],
            "fn_count": cand["fn_count"],
            "other_count": cand["other_count"],
            "fn_rate": cand["fn_rate"],
            "other_rate": cand["other_rate"],
            "support_diff": cand["support_diff"],
            "posterior": cand["posterior"],
            "bayes_factor": cand["bayes_factor"],
            "bonus_points": bonus_points,
            "predicates": cand["predicates"],
        }
        rule["activation_conditions"] = _reverse_bonus_rule_conditions(rule, 0.62)
        rule["activation_conditions_text"] = " / ".join(rule["activation_conditions"])
        rule["activation_description"] = _reverse_bonus_rule_description(rule, 0.62)
        rules.append(rule)
        if len(rules) >= 6:
            break

    result = {
        "gate": {"deal_source": "その他", "customer_type": "新規先"},
        "n_cases": len(gate_cases),
        "n_fn": n_fn,
        "n_other": n_other,
        "threshold": threshold,
        "activation_threshold": 0.62,
        "rules": rules,
    }
    _REVERSE_BONUS_CACHE.update({"db_mtime": db_mtime, "result": result})
    return result


def _scope_summary(cases: list[dict]) -> list[dict[str, Any]]:
    threshold = 71.0
    rows: list[dict[str, Any]] = []
    for case in cases:
        score = _reverse_bonus_predict_score(case)
        if score is None:
            continue
        actual_success = case.get("final_status") in SUCCESS_STATUSES
        pred_success = score >= threshold
        rows.append({
            "case": case,
            "actual_success": actual_success,
            "pred_success": pred_success,
            "sales_dept": _norm_str(_get_raw(case, "sales_dept")) or "未設定",
            "industry_major": _norm_str(_get_raw(case, "industry_major")) or "未設定",
        })
    return rows


def _best_patch_scopes(cases: list[dict], key: str, top_k: int = 3) -> list[tuple[str, list[dict]]]:
    rows = _scope_summary(cases)
    agg: dict[str, dict[str, int]] = {}
    for row in rows:
        scope = row[key]
        item = agg.setdefault(scope, {"n": 0, "success": 0, "fn": 0, "fp": 0})
        item["n"] += 1
        if row["actual_success"]:
            item["success"] += 1
            if not row["pred_success"]:
                item["fn"] += 1
        elif row["pred_success"]:
            item["fp"] += 1
    scored: list[tuple[float, str]] = []
    for scope, item in agg.items():
        if item["n"] < 40 or item["success"] < 20:
            continue
        fn_rate = item["fn"] / item["success"] if item["success"] else 0.0
        scored.append((fn_rate, scope))
    scored.sort(reverse=True)
    out: list[tuple[str, list[dict]]] = []
    for _, scope in scored[:top_k]:
        filtered = [row["case"] for row in rows if row[key] == scope]
        out.append((scope, filtered))
    return out


def _enumerate_fp0_patches(cases: list[dict], top_k: int = 30) -> list[dict[str, Any]]:
    n_success = sum(1 for c in cases if c.get("final_status") in SUCCESS_STATUSES)
    n_loss = len(cases) - n_success
    atoms = _build_atoms(cases)
    success_bits, loss_bits = _build_bitsets(cases, atoms)

    candidates: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for combo in combinations(range(len(atoms)), size):
            m = _pattern_metrics(combo, atoms, success_bits, loss_bits, n_success, n_loss)
            if m.get("invalid"):
                continue
            if m["loss_count"] != 0 or m["success_count"] < 1:
                continue
            candidates.append(m)

    candidates.sort(
        key=lambda x: (
            x["success_count"],
            x["success_coverage"],
            x["precision"],
            x["support_diff"],
            -x["n_items"],
        ),
        reverse=True,
    )
    return candidates[:top_k]


def _fp0_candidate_category(candidate: dict[str, Any], n_success: int) -> str:
    adoption_cutoff = max(50, int(n_success * 0.10))
    if candidate.get("loss_count", 0) == 0 and candidate.get("success_count", 0) >= adoption_cutoff:
        return "採用候補"
    return "補助ルール"


def _fp0_candidate_description(candidate: dict[str, Any], scope_label: str = "") -> str:
    parts = []
    if scope_label:
        parts.append(f"{scope_label}に限定した局所 gate")
    if candidate.get("loss_count", 0) == 0:
        parts.append("失注への誤爆を0件に保ったまま成約を救う")
    else:
        parts.append("失注も拾うため採用候補にはしない")
    if candidate.get("success_count") is not None:
        parts.append(f"成約 {int(candidate['success_count'])}件")
    if candidate.get("success_coverage") is not None:
        parts.append(f"成約内割合 {float(candidate['success_coverage']) * 100:.1f}%")
    if candidate.get("precision") is not None:
        parts.append(f"純度 {float(candidate['precision']) * 100:.1f}%")
    return " / ".join(parts)


def _fp0_atom_to_jp(atom_label: str) -> str:
    mapping = {
        "sales": "売上高",
        "nenshu": "売上高",
        "gross_profit": "売上総利益",
        "op_profit": "営業利益",
        "ord_profit": "経常利益",
        "net_income": "純利益",
        "machines": "機械・設備",
        "other_assets": "その他資産",
        "total_assets": "総資産",
        "net_assets": "純資産",
        "rent": "賃料",
        "depreciation": "減価償却費",
        "rent_expense": "賃借料",
        "bank_credit": "銀行借入",
        "lease_credit": "リース借入",
        "contracts": "契約件数",
        "lease_term": "リース期間",
        "acquisition_cost": "導入コスト",
        "lease_asset_score": "物件適性スコア",
        "op_margin": "営業利益率",
        "gross_margin": "売上総利益率",
        "bank_sales_ratio": "売上に対する銀行借入比率",
        "lease_sales_ratio": "売上に対するリース比率",
        "debt_sales_ratio": "売上に対する負債比率",
        "dep_profit_ratio": "営業利益に対する減価償却比率",
        "equity_ratio": "自己資本比率",
        "q_risk": "Q_risk",
        "mahalanobis_distance": "マハラノビス距離",
        "customer_type": "顧客区分",
        "deal_source": "商談ソース",
        "industry_major": "業種大分類",
        "industry_sub": "業種小分類",
        "sales_dept": "営業部",
        "grade": "格付",
        "contract_type": "契約種別",
        "competitor": "競合状況",
    }
    if "<=" in atom_label or ">=" in atom_label:
        left, right = atom_label.split(" ", 1)
        key = left.strip()
        op, value = right.split(" ", 1)
        jp_key = mapping.get(key, key)
        if op == "<=":
            return f"{jp_key} が低い（{value}以下）"
        if op == ">=":
            return f"{jp_key} が高い（{value}以上）"
        return f"{jp_key} が {value}"
    if "=" in atom_label:
        key, value = atom_label.split("=", 1)
        jp_key = mapping.get(key.strip(), key.strip())
        if value.strip() in {"None", "null", "空欄"}:
            return f"{jp_key} が空欄"
        return f"{jp_key} が {value.strip()}"
    return mapping.get(atom_label, atom_label)


def _fp0_patch_explanation(candidate: dict[str, Any], scope_label: str = "") -> dict[str, str]:
    atoms = candidate.get("items") or []
    jp_atoms = [_fp0_atom_to_jp(atom) for atom in atoms]
    title = "営業部別・特異成約パターン検知"
    if scope_label:
        title = f"{title} - {scope_label}"
    explanation = "、".join(jp_atoms) if jp_atoms else "複数条件の組み合わせ"
    explanation = (
        f"この条件が強いのは、{explanation} という成約に効きやすい特徴が同時に揃っており、"
        f"失注側には現れていないためです。"
    )
    action = "この案件はデータ不足を補う Soul因子 が検出されました。ベイズ加点を推奨します。"
    return {
        "title": title,
        "explanation": explanation,
        "action": action,
    }


def mine_fp0_patch_candidates(force_recompute: bool = False) -> dict[str, Any] | None:
    db_mtime = _db_mtime()
    if not force_recompute and _PATCH_CACHE.get("result") is not None and _PATCH_CACHE.get("db_mtime") == db_mtime:
        return _PATCH_CACHE["result"]

    cases = _build_cases()
    if not cases:
        return None

    global_candidates = _enumerate_fp0_patches(cases, top_k=30)
    global_adoption_candidates: list[dict[str, Any]] = []
    global_auxiliary_rules: list[dict[str, Any]] = []
    for cand in global_candidates:
        enriched = dict(cand)
        enriched["category"] = _fp0_candidate_category(cand, sum(1 for c in cases if c.get("final_status") in SUCCESS_STATUSES))
        enriched["description"] = _fp0_candidate_description(cand, "全体")
        enriched.update(_fp0_patch_explanation(cand, "全体"))
        if enriched["category"] == "採用候補":
            global_adoption_candidates.append(enriched)
        else:
            global_auxiliary_rules.append(enriched)
    top_depts = _best_patch_scopes(cases, "sales_dept", top_k=3)
    top_inds = _best_patch_scopes(cases, "industry_major", top_k=3)

    dept_results: list[dict[str, Any]] = []
    for scope, scope_cases in top_depts:
        scope_candidates = _enumerate_fp0_patches(scope_cases, top_k=10)
        dept_results.append({
            "scope": scope,
            "n_cases": len(scope_cases),
            "candidates": [
                {
                    **cand,
                    "category": "補助ルール",
                    "description": _fp0_candidate_description(cand, f"{scope}"),
                    **_fp0_patch_explanation(cand, scope),
                }
                for cand in scope_candidates
            ],
        })

    industry_results: list[dict[str, Any]] = []
    for scope, scope_cases in top_inds:
        scope_candidates = _enumerate_fp0_patches(scope_cases, top_k=10)
        industry_results.append({
            "scope": scope,
            "n_cases": len(scope_cases),
            "candidates": [
                {
                    **cand,
                    "category": "補助ルール",
                    "description": _fp0_candidate_description(cand, f"{scope}"),
                    **_fp0_patch_explanation(cand, scope),
                }
                for cand in scope_candidates
            ],
        })

    result = {
        "n_cases": len(cases),
        "global_candidates": [
            {
                **cand,
                "category": _fp0_candidate_category(cand, sum(1 for c in cases if c.get("final_status") in SUCCESS_STATUSES)),
                "description": _fp0_candidate_description(cand, "全体"),
                **_fp0_patch_explanation(cand, "全体"),
            }
            for cand in global_candidates
        ],
        "adoption_candidates": global_adoption_candidates,
        "auxiliary_rules": global_auxiliary_rules,
        "dept_results": dept_results,
        "industry_results": industry_results,
        "criteria": {
            "loss_count_max": 0,
            "success_count_min": 1,
            "source": "all_input_fields_plus_reference_vars",
        },
    }
    _PATCH_CACHE.update({"db_mtime": db_mtime, "result": result})
    return result


def _fp0_atom_hit(case: dict, atom: dict[str, Any]) -> bool:
    if atom["kind"] == "categorical":
        key = atom["key"]
        if key == "grade":
            current = _grade_norm(_get_raw(case, "grade"))
        else:
            current = _norm_str(_get_raw(case, key))
        return current == atom["threshold"]
    if atom["key"] == "sales":
        value = _get_float(case, "nenshu")
    elif atom["key"] == "bank_credit":
        value = _get_float(case, "bank_credit")
    elif atom["key"] == "bank_ratio":
        value = _reverse_bonus_bank_ratio(case)
    elif atom["key"] == "rent":
        value = _get_float(case, "rent")
    elif atom["key"] == "acquisition_cost":
        value = _get_float(case, "acquisition_cost")
    elif atom["key"] == "dep_profit_ratio":
        value = _dep_profit_ratio(case)
    elif atom["key"] == "lease_sales_ratio":
        value = _lease_sales_ratio(case)
    elif atom["key"] == "gross_margin":
        value = _gross_margin(case)
    elif atom["key"] == "op_margin":
        value = _op_margin(case)
    elif atom["key"] == "mahalanobis_distance":
        value = _mahalanobis_distance(case)
    else:
        value = _get_float(case, atom["key"])
    if value is None or not math.isfinite(float(value)):
        return False
    if atom["op"] == ">=":
        return float(value) >= float(atom["threshold"])
    return float(value) <= float(atom["threshold"])


def build_fp0_patch_note(
    case: dict,
    force_recompute: bool = False,
    candidate_buckets: tuple[str, ...] = ("adoption_candidates",),
) -> dict[str, Any] | None:
    analysis = mine_fp0_patch_candidates(force_recompute=force_recompute)
    if not analysis:
        return None

    best: dict[str, Any] | None = None
    for bucket_name in candidate_buckets:
        for cand in analysis.get(bucket_name) or []:
            predicates = cand.get("predicates") or []
            try:
                if not predicates:
                    continue
                if not all(_fp0_atom_hit(case, atom) for atom in predicates):
                    continue
            except Exception:
                continue
            if best is None or (
                cand.get("success_count", 0),
                cand.get("precision", 0.0),
                cand.get("success_coverage", 0.0),
            ) > (
                best.get("success_count", 0),
                best.get("precision", 0.0),
                best.get("success_coverage", 0.0),
            ):
                best = cand

    if best is None:
        return None

    return {
        "source": "fp0_patch",
        "title": best.get("title") or "営業部別・特異成約パターン検知",
        "pattern": best.get("pattern"),
        "category": best.get("category"),
        "success_count": best.get("success_count"),
        "loss_count": best.get("loss_count"),
        "success_coverage": best.get("success_coverage"),
        "precision": best.get("precision"),
        "description": best.get("description") or "",
        "explanation": best.get("explanation") or "",
        "action": best.get("action") or "この案件はデータ不足を補う Soul因子 が検出されました。ベイズ加点を推奨します。",
        "scope": best.get("category"),
        "note": (
            f"{best.get('title') or '営業部別・特異成約パターン検知'}\n"
            f"{best.get('explanation') or ''}\n"
            f"{best.get('action') or ''}"
        ).strip(),
    }


def build_reverse_bayes_bonus(case: dict, force_recompute: bool = False) -> dict[str, Any] | None:
    analysis = mine_reverse_bayes_bonus(force_recompute=force_recompute)
    if not analysis:
        return None
    if not _reverse_bonus_gate(case):
        return None
    rules = analysis.get("rules") or []
    matched: list[dict[str, Any]] = []
    for rule in rules:
        predicates = rule.get("predicates") or []
        try:
            if all(_reverse_bonus_atom_hit(case, atom) for atom in predicates):
                matched.append(rule)
        except Exception:
            continue
    if not matched:
        return None

    best = max(
        matched,
        key=lambda x: (
            x.get("posterior", 0.0),
            x.get("fn_count", 0),
            -x.get("other_count", 0),
            -x.get("n_items", 0),
        ),
    )
    posterior = float(best.get("posterior") or 0.0)
    activation_threshold = float(analysis.get("activation_threshold") or 0.62)
    if posterior < activation_threshold:
        return None

    return {
        "source": "reverse_bayes",
        "gate": analysis.get("gate"),
        "n_cases": analysis.get("n_cases"),
        "n_fn": analysis.get("n_fn"),
        "n_other": analysis.get("n_other"),
        "rule": {
            "pattern": best.get("pattern"),
            "fn_count": best.get("fn_count"),
            "other_count": best.get("other_count"),
            "posterior": round(posterior, 3),
            "bayes_factor": round(float(best.get("bayes_factor") or 0.0), 3),
            "bonus_points": int(best.get("bonus_points") or 0),
            "activation_conditions": best.get("activation_conditions") or _reverse_bonus_rule_conditions(best, activation_threshold),
            "activation_conditions_text": best.get("activation_conditions_text") or " / ".join(
                best.get("activation_conditions") or _reverse_bonus_rule_conditions(best, activation_threshold)
            ),
            "activation_description": best.get("activation_description") or _reverse_bonus_rule_description(best, activation_threshold),
        },
        "bonus_points": int(best.get("bonus_points") or 0),
        "posterior": round(posterior, 3),
        "activation_threshold": activation_threshold,
        "activation_conditions": best.get("activation_conditions") or _reverse_bonus_rule_conditions(best, activation_threshold),
        "activation_conditions_text": best.get("activation_conditions_text") or " / ".join(
            best.get("activation_conditions") or _reverse_bonus_rule_conditions(best, activation_threshold)
        ),
        "activation_description": best.get("activation_description") or _reverse_bonus_rule_description(best, activation_threshold),
    }
