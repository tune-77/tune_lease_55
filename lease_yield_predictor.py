"""
リース利回り推測モジュール（試作版）。

注: 結果登録画面（components/form_status.py）で保存される `final_rate` が
    成約時の獲得利率（= contract_rate）として past_cases に蓄積される。
    calibrate_from_history() でその実績を集計し、スプレッドの精度を確認できる。

推測式:
  predicted_yield(%) = funding_rate(月, 期間)   # 自社調達金利
                     + asset_spread(資産種別, 期間)
                     + grade_spread(借手格付)
                     + risk_adjustment(借手スコア)

使い方:
  import sqlite3
  from lease_yield_predictor import predict_yield

  conn = sqlite3.connect("data/lease_data.db")
  result = predict_yield(conn, {
      "year_month": "2026-04",
      "lease_term_months": 60,
      "lease_asset_id": "medical",
      "grade": "②4-6 (標準)",
      "borrower_score": 96.2,
  })
  print(result)
"""

import json
import sqlite3

# ── 資産スプレッドテーブル（期間別、単位: %） ──────────────────────────────
# キー: (asset_id_prefix, term_years) → spread_pct
# 根拠: past_cases の rate_diff 分布（-0.04〜+0.5%程度）を踏まえた暫定値
_ASSET_SPREADS: dict[tuple[str, int], float] = {
    # 医療機器: 残存価値安定、規制市場。期間が延びると残価リスク増
    ("medical",      1): 0.25, ("medical",      3): 0.30, ("medical",      5): 0.35,
    ("medical",      7): 0.40, ("medical",     10): 0.50,
    # IT機器: 陳腐化リスク大
    ("it",           1): 0.35, ("it",           3): 0.50, ("it",           5): 0.65,
    ("it",           7): 0.75, ("it",          10): 0.90,
    ("pc",           1): 0.35, ("pc",           3): 0.50, ("pc",           5): 0.65,
    ("pc",           7): 0.75, ("pc",          10): 0.90,
    # 車両
    ("vehicle",      1): 0.28, ("vehicle",      3): 0.32, ("vehicle",      5): 0.38,
    ("vehicle",      7): 0.45, ("vehicle",     10): 0.55,
    ("car",          1): 0.28, ("car",          3): 0.32, ("car",          5): 0.38,
    ("car",          7): 0.45, ("car",         10): 0.55,
    # 工作機械
    ("machinery",    1): 0.30, ("machinery",    3): 0.38, ("machinery",    5): 0.45,
    ("machinery",    7): 0.52, ("machinery",   10): 0.60,
    # 建設機械
    ("construction", 1): 0.32, ("construction", 3): 0.40, ("construction", 5): 0.48,
    ("construction", 7): 0.55, ("construction",10): 0.65,
}
_ASSET_DEFAULT: dict[int, float] = {1: 0.32, 3: 0.42, 5: 0.50, 7: 0.58, 10: 0.68}

# ── 格付スプレッドテーブル（単位: %） ──────────────────────────────────────
# 入力 grade は "①1-3 (優良)" / "②4-6 (標準)" / "③7-9 (注意)" など
_GRADE_SPREADS: dict[str, float] = {
    "s":  -0.10,
    "①": -0.10,  # 優良
    "a":   0.10,
    "②":  0.25,  # 標準
    "b":   0.25,
    "③":  0.55,  # 注意
    "c":   0.55,
    "④":  0.90,  # 要注意
    "d":   0.90,
}
_GRADE_DEFAULT = 0.30

# ── 有効な期間リスト ──────────────────────────────────────────────────────
_VALID_TERMS = [1, 3, 5, 7, 10]


def _nearest_term(term_years: float) -> int:
    """リース期間（年）を最寄りの有効期間に丸める。"""
    return min(_VALID_TERMS, key=lambda t: abs(t - term_years))


def get_funding_rate(
    conn: sqlite3.Connection,
    year_month: str,
    term_years: int,
) -> tuple[float, bool]:
    """
    指定月・期間の自社調達金利を返す。
    該当月が未登録の場合は直近過去月に遡る（フォールバック）。

    Returns:
        (rate_pct, fallback_used): 金利（%）とフォールバック有無
    """
    cur = conn.execute(
        """
        SELECT year_month, rate_pct
        FROM funding_rates
        WHERE term_years = ? AND year_month <= ?
        ORDER BY year_month DESC
        LIMIT 1
        """,
        (term_years, year_month),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"funding_rates に {year_month} 以前の {term_years}Y データがありません。"
            " import_funding_rates.py でデータを投入してください。"
        )
    found_ym, rate = row
    fallback = found_ym != year_month
    return rate, fallback


def get_asset_spread(asset_id: str, term_years: int) -> float:
    """資産種別と期間からスプレッドを返す（%）。"""
    key = (asset_id.lower(), term_years)
    if key in _ASSET_SPREADS:
        return _ASSET_SPREADS[key]
    # prefix部分一致（例: "medical_device" → "medical"）
    for prefix in _ASSET_SPREADS:
        if prefix[0] and asset_id.lower().startswith(prefix[0]):
            alt_key = (prefix[0], term_years)
            if alt_key in _ASSET_SPREADS:
                return _ASSET_SPREADS[alt_key]
    return _ASSET_DEFAULT.get(term_years, 0.45)


def get_grade_spread(grade: str) -> float:
    """借手格付からスプレッドを返す（%）。"""
    g = grade.strip().lower()
    for key, spread in _GRADE_SPREADS.items():
        if g.startswith(key):
            return spread
    return _GRADE_DEFAULT


def get_risk_adjustment(borrower_score: float) -> float:
    """
    借手スコア（0〜100）からリスク補正を返す（%）。
    高スコア: 割引、低スコア: 上乗せ。
    """
    if borrower_score >= 90:
        return -0.10
    if borrower_score >= 70:
        return 0.00
    if borrower_score >= 50:
        return 0.20
    return 0.40


def predict_yield(conn: sqlite3.Connection, inputs: dict) -> dict:
    """
    利回りを推測する。

    Args:
        conn: lease_data.db への接続
        inputs: {
            "year_month":        "YYYY-MM",          # 審査年月
            "lease_term_months": int,                 # 期間（月）
            "lease_asset_id":    str,                 # 資産種別ID
            "grade":             str,                 # 借手格付
            "borrower_score":    float,               # 借手スコア（0-100）
        }

    Returns: {
        "predicted_yield":  float,  # 推測利回り（%）
        "breakdown": {
            "base":   float,        # 自社調達金利
            "asset":  float,        # 資産スプレッド
            "grade":  float,        # 格付スプレッド
            "risk":   float,        # リスク補正
        },
        "term_years_used":  int,    # 使用した期間（最寄り値）
        "fallback_used":    bool,   # 過去月の金利を流用したか
        "fallback_note":    str,    # フォールバック説明
    }
    """
    year_month      = inputs["year_month"]
    lease_term_months = inputs["lease_term_months"]
    asset_id        = inputs.get("lease_asset_id", "other")
    grade           = inputs.get("grade", "")
    borrower_score  = float(inputs.get("borrower_score", 70.0))

    term_years_raw  = lease_term_months / 12
    term_years      = _nearest_term(term_years_raw)

    base, fallback = get_funding_rate(conn, year_month, term_years)
    asset  = get_asset_spread(asset_id, term_years)
    grade_ = get_grade_spread(grade)
    risk   = get_risk_adjustment(borrower_score)

    total = round(base + asset + grade_ + risk, 4)

    fallback_note = ""
    if fallback:
        cur = conn.execute(
            "SELECT year_month FROM funding_rates WHERE term_years=? AND year_month<=? ORDER BY year_month DESC LIMIT 1",
            (term_years, year_month),
        )
        row = cur.fetchone()
        fallback_note = f"指定月 {year_month} のデータ未登録のため {row[0]} の値を使用"

    return {
        "predicted_yield": total,
        "breakdown": {
            "base":  base,
            "asset": asset,
            "grade": grade_,
            "risk":  risk,
        },
        "term_years_used": term_years,
        "fallback_used":   fallback,
        "fallback_note":   fallback_note,
    }


def calibrate_from_history(conn: sqlite3.Connection) -> dict:
    """
    past_cases の実績（成約 + final_rate > 0）から資産・期間別の実績スプレッドを集計する。

    データが十分に貯まったら、この結果で _ASSET_SPREADS を校正できる。
    - 10件以上: 参考値として使い始める目安
    - 30件以上: スプレッドテーブルの上書きを検討

    Returns: {
        "n_cases":             int,    # 集計に使った案件数
        "n_skipped":           int,    # データ不足などでスキップした件数
        "spread_by_asset_term": {      # (資産/期間) → 実績スプレッドの統計
            "medical/5Y": {
                "count":      int,
                "avg_spread": float,
                "min_spread": float,
                "max_spread": float,
            }, ...
        },
    }
    """
    rows = conn.execute(
        "SELECT data FROM past_cases WHERE final_status = '成約'"
    ).fetchall()

    spread_data: dict[str, list[float]] = {}
    skipped = 0

    for (data_json,) in rows:
        try:
            data = json.loads(data_json or "{}")
        except Exception:
            skipped += 1
            continue

        final_rate = data.get("final_rate", 0)
        if not final_rate or final_rate <= 0:
            skipped += 1
            continue

        ts = data.get("timestamp", "")
        year_month = ts[:7] if len(ts) >= 7 else None
        if not year_month:
            skipped += 1
            continue

        inputs = data.get("inputs", {})
        lease_term_months = inputs.get("lease_term", 60)
        asset_id = inputs.get("lease_asset_id", "other")
        term_years = _nearest_term(lease_term_months / 12)

        try:
            base_rate, _ = get_funding_rate(conn, year_month, term_years)
        except ValueError:
            skipped += 1
            continue

        actual_spread = final_rate - base_rate
        key = f"{asset_id}/{term_years}Y"
        spread_data.setdefault(key, []).append(actual_spread)

    summary = {}
    for key, spreads in spread_data.items():
        summary[key] = {
            "count":      len(spreads),
            "avg_spread": round(sum(spreads) / len(spreads), 4),
            "min_spread": round(min(spreads), 4),
            "max_spread": round(max(spreads), 4),
        }

    return {
        "n_cases":              sum(len(v) for v in spread_data.values()),
        "n_skipped":            skipped,
        "spread_by_asset_term": summary,
    }
