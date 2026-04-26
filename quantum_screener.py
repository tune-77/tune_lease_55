#!/usr/bin/env python3
"""
quantum_screener.py
===================
DB の未使用変数と失注の相関係数・p値を CSV に出力するスクリーニングツール。

使用方法:
  python3 quantum_screener.py              # 相関分析 CSV を stdout に出力
  python3 quantum_screener.py --out FILE   # CSV をファイルに保存
  python3 quantum_screener.py --suggest    # quantum_config.json に候補ペアを追記（SC.5）
"""
from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

DB_PATH = Path("data/lease_data.db")

# 現行量子モデルで使用していない変数（スクリーニング対象）
UNUSED_VARS: list[str] = [
    "nenshu",
    "bank_credit",
    "other_assets",
    "gross_profit",
    "rent",
    "dep_expense",
    "rent_expense",
    "lease_credit",
    "contracts",
    "acquisition_cost",
]


def _pearson_r_pvalue(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson 相関係数と p値（t 分布近似）を返す。"""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    if _HAS_NUMPY:
        arr_x = np.array(x, dtype=float)
        arr_y = np.array(y, dtype=float)
        sx = float(np.std(arr_x, ddof=1))
        sy = float(np.std(arr_y, ddof=1))
        if sx == 0 or sy == 0:
            return 0.0, 1.0
        r = float(np.corrcoef(arr_x, arr_y)[0, 1])
    else:
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if den_x == 0 or den_y == 0:
            return 0.0, 1.0
        r = num / (den_x * den_y)

    r = max(-1.0, min(1.0, r))

    # p値: t = r * sqrt(n-2) / sqrt(1-r^2)、自由度 n-2 の t 分布
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    # 正規近似（|t| が大きい場合は精度十分）
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return r, max(0.0, min(1.0, p_value))


def _normal_cdf(z: float) -> float:
    """標準正規分布の CDF（近似）"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class QuantumScreener:
    """未使用変数と失注の相関を分析するクラス。

    Args:
        db_path: SQLite DB パス（省略時は data/lease_data.db）
        records: テスト用インメモリレコードリスト（DB の代わりに使用）
        vars_to_screen: スクリーニング対象変数リスト（省略時は UNUSED_VARS）
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        records: list[dict[str, Any]] | None = None,
        vars_to_screen: list[str] | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._records = records  # テスト用インメモリ注入
        self.vars_to_screen = vars_to_screen or UNUSED_VARS

    def _load_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if not self._db_path.exists():
            return []
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT data, final_status FROM past_cases"
            ).fetchall()
        finally:
            conn.close()
        result = []
        for raw, status in rows:
            try:
                d = json.loads(raw)
                d["final_status"] = status or d.get("final_status", "")
                result.append(d)
            except Exception:
                pass
        return result

    def compute_correlations(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """未使用変数と失注の相関係数・p値を DataFrame で返す。

        Returns:
            columns: variable, correlation, p_value, n
        """
        import pandas as pd  # pandas はオプション依存

        records = self._load_records()
        rows = []

        for var in self.vars_to_screen:
            x_vals: list[float] = []
            y_vals: list[float] = []  # 1=失注, 0=成約

            for rec in records:
                inputs = rec.get("inputs", rec)
                raw = inputs.get(var)
                if raw is None:
                    continue
                try:
                    x = float(raw)
                except (TypeError, ValueError):
                    continue
                status = str(rec.get("final_status", ""))
                y = 1.0 if status == "失注" else 0.0
                x_vals.append(x)
                y_vals.append(y)

            n = len(x_vals)
            if n < 3:
                rows.append({"variable": var, "correlation": 0.0, "p_value": 1.0, "n": n})
                continue

            r, p = _pearson_r_pvalue(x_vals, y_vals)
            rows.append({
                "variable": var,
                "correlation": round(r, 6),
                "p_value": round(p, 6),
                "n": n,
            })

        df = pd.DataFrame(rows, columns=["variable", "correlation", "p_value", "n"])
        return df.sort_values("p_value").reset_index(drop=True)

    def to_csv(self, path: Path | str | None = None) -> str:
        """相関分析結果を CSV 文字列で返し、path が指定されればファイルにも保存する。"""
        df = self.compute_correlations()
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_text = buf.getvalue()
        if path:
            Path(path).write_text(csv_text, encoding="utf-8")
        return csv_text

    def suggest_pairs(self) -> list[dict[str, Any]]:
        """相関が強い変数から新規ペア候補を生成する（SC.5 用）。

        Returns:
            [{"var_a": str, "var_b": str, "weight": float}, ...]
        """
        df = self.compute_correlations()
        significant = df[df["p_value"] < 0.1].head(5)
        candidates = []
        vars_sig = list(significant["variable"])
        current_quantum_vars = ["op_profit", "depreciation", "machines", "net_income", "ord_profit"]
        for new_var in vars_sig:
            for base_var in current_quantum_vars[:2]:
                candidates.append({
                    "var_a": base_var,
                    "var_b": new_var,
                    "weight": 1.0,
                    "source": "quantum_screener",
                })
        return candidates


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="未使用変数スクリーニング")
    parser.add_argument("--out", metavar="FILE", help="CSV 出力先ファイルパス")
    parser.add_argument("--suggest", action="store_true", help="quantum_config.json に候補ペアを追記")
    args = parser.parse_args()

    screener = QuantumScreener()

    if args.suggest:
        candidates = screener.suggest_pairs()
        cfg_path = Path("data/quantum_config.json")
        cfg: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        cfg.setdefault("candidate_pairs", [])
        existing = {(c["var_a"], c["var_b"]) for c in cfg["candidate_pairs"]}
        added = 0
        for cand in candidates:
            key = (cand["var_a"], cand["var_b"])
            if key not in existing:
                cfg["candidate_pairs"].append(cand)
                existing.add(key)
                added += 1
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"候補ペア {added} 件を {cfg_path} に追記しました。", file=sys.stderr)
        return

    out_path = args.out
    csv_text = screener.to_csv(path=out_path)
    if out_path:
        print(f"CSV を {out_path} に保存しました。", file=sys.stderr)
    else:
        print(csv_text, end="")


if __name__ == "__main__":
    main()
