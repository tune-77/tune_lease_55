"""利回り推測ロジックのテスト（最小2ケース）"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lease_yield_predictor import predict_yield


@pytest.fixture
def conn():
    """インメモリDB にサンプルデータを用意する。"""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE funding_rates (
            year_month TEXT NOT NULL,
            term_years INTEGER NOT NULL,
            rate_pct REAL NOT NULL,
            source TEXT DEFAULT 'manual',
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (year_month, term_years)
        )
    """)
    db.executemany(
        "INSERT INTO funding_rates (year_month, term_years, rate_pct) VALUES (?,?,?)",
        [
            ("2026-04", 1, 0.60),
            ("2026-04", 3, 0.98),
            ("2026-04", 5, 1.28),
            ("2026-04", 7, 1.52),
            ("2026-03", 5, 1.25),
        ],
    )
    db.commit()
    yield db
    db.close()


def test_normal_case(conn):
    """正常系: 月・期間が一致する場合、合算値が式通りになること。"""
    result = predict_yield(conn, {
        "year_month":        "2026-04",
        "lease_term_months": 60,        # → 5年
        "lease_asset_id":    "medical",
        "grade":             "②4-6 (標準)",
        "borrower_score":    96.2,
    })

    bk = result["breakdown"]
    assert result["term_years_used"] == 5
    assert not result["fallback_used"]
    # 合計が内訳の和と一致（浮動小数誤差を許容）
    expected = bk["base"] + bk["asset"] + bk["grade"] + bk["risk"]
    assert abs(result["predicted_yield"] - round(expected, 4)) < 1e-9
    # 調達金利は 1.28%
    assert bk["base"] == pytest.approx(1.28)
    # 医療機器5年スプレッドは 0.35%
    assert bk["asset"] == pytest.approx(0.35)
    # ②格付スプレッドは 0.25%
    assert bk["grade"] == pytest.approx(0.25)
    # スコア96.2 → -0.10%
    assert bk["risk"] == pytest.approx(-0.10)
    assert result["predicted_yield"] == pytest.approx(1.78)


def test_fallback_case(conn):
    """フォールバック系: 指定月のデータが無い場合、過去月を使うこと。"""
    result = predict_yield(conn, {
        "year_month":        "2026-05",  # 未登録月
        "lease_term_months": 60,         # → 5年
        "lease_asset_id":    "medical",
        "grade":             "②4-6 (標準)",
        "borrower_score":    70.0,
    })

    assert result["fallback_used"] is True
    assert "2026-04" in result["fallback_note"]
    # 過去月（2026-04）の 5Y レート 1.28% を使うこと
    assert result["breakdown"]["base"] == pytest.approx(1.28)
