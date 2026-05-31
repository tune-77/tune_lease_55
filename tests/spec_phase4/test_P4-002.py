"""
P4-002 EDINET データ自動収集 — 単体テスト（AC-1101〜AC-1110）
"""
from __future__ import annotations

import io
import sqlite3
import time
import zipfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ヘルパー: テスト用インメモリ DB & edinet_collector をパス設定込みで呼ぶ
# ---------------------------------------------------------------------------

def _make_in_memory_db_path(tmp_path):
    db_path = str(tmp_path / "test_edinet.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edinet_cache (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            corporate_number    TEXT    NOT NULL,
            fiscal_year         INTEGER NOT NULL,
            edinet_code         TEXT,
            nenshu              REAL,
            operating_profit    REAL,
            net_income          REAL,
            total_assets        REAL,
            equity_ratio        REAL,
            raw_json            TEXT,
            fetched_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(corporate_number, fiscal_year)
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def _make_xbrl_zip(
    nenshu_yen: int = 1_234_000_000,
    op_profit_yen: int = 123_000_000,
    net_income_yen: int = 45_000_000,
    assets_yen: int = 3_000_000_000,
    equity_ratio_pct: float | None = 38.5,
) -> bytes:
    """財務数値を埋め込んだ最小限の XBRL zip を生成する。"""
    equity_tag = (
        f'<jppfs_cor:EquityToAssetRatio contextRef="CurrentYearDuration">'
        f"{equity_ratio_pct}</jppfs_cor:EquityToAssetRatio>"
        if equity_ratio_pct is not None
        else ""
    )
    xbrl_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrl
  xmlns:jppfs_cor="http://disclosure.edinet-api.go.jp/taxonomy/jppfs/2019-11-01/jppfs_cor"
  xmlns:xbrli="http://www.xbrl.org/2003/instance"
>
  <jppfs_cor:NetSales contextRef="CurrentYearConsolidatedDuration">{nenshu_yen}</jppfs_cor:NetSales>
  <jppfs_cor:OperatingIncome contextRef="CurrentYearConsolidatedDuration">{op_profit_yen}</jppfs_cor:OperatingIncome>
  <jppfs_cor:ProfitLossAttributableToOwnersOfParent contextRef="CurrentYearConsolidatedDuration">{net_income_yen}</jppfs_cor:ProfitLossAttributableToOwnersOfParent>
  <jppfs_cor:Assets contextRef="CurrentYearConsolidatedInstant">{assets_yen}</jppfs_cor:Assets>
  {equity_tag}
</xbrl>
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("XBRL_DOC/jpcrp_fr.xbrl", xbrl_content.encode())
    return buf.getvalue()


def _make_companies_response(corporate_number: str, edinet_code: str = "E12345") -> dict:
    return {
        "results": [
            {"corporateNumber": corporate_number, "edinetCode": edinet_code}
        ]
    }


def _make_documents_response(edinet_code: str, doc_id: str = "S100XXXX") -> dict:
    return {
        "results": [
            {
                "edinetCode": edinet_code,
                "docID": doc_id,
                "formCode": "030000",
            }
        ]
    }


def _mock_requests_get(
    corporate_number: str = "1234567890123",
    edinet_code: str = "E12345",
    doc_id: str = "S100XXXX",
    xbrl_zip: bytes | None = None,
    timeout: bool = False,
    http_error_status: int | None = None,
):
    """
    requests.get をモックする関数を返す。
    URL に応じて適切なレスポンスを返す。
    """
    if xbrl_zip is None:
        xbrl_zip = _make_xbrl_zip()

    def _side_effect(url, **kwargs):
        if timeout:
            raise requests.exceptions.Timeout("mocked timeout")
        if http_error_status:
            resp = MagicMock()
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                f"HTTP {http_error_status}"
            )
            return resp

        resp = MagicMock()
        resp.raise_for_status.return_value = None

        if "companies.json" in url:
            resp.json.return_value = _make_companies_response(corporate_number, edinet_code)
        elif "documents.json" in url:
            resp.json.return_value = _make_documents_response(edinet_code, doc_id)
        elif f"documents/{doc_id}" in url:
            resp.content = xbrl_zip
        else:
            resp.json.return_value = {"results": []}
        return resp

    return _side_effect


# ---------------------------------------------------------------------------
# インポート
# ---------------------------------------------------------------------------

import requests  # noqa: E402 (after helper defs for type hints)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from edinet_collector import (
    fetch_edinet_financials,
    resolve_edinet_code,
    _parse_xbrl_financials,
    _validate_corporate_number,
    _get_db_connection,
    _is_company_cache_fresh,
    _lookup_edinet_code_from_cache,
    _write_company_list_cache,
)


# ---------------------------------------------------------------------------
# test_1101: AC-1101 — モックAPIで正常取得 → success=True, source="edinet"
# ---------------------------------------------------------------------------

def test_1101_normal_fetch(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    with patch("requests.get", side_effect=_mock_requests_get()):
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, db_path=db_path
        )
    assert result["success"] is True
    assert result["source"] == "edinet"
    assert isinstance(result["nenshu"], float)
    assert result["nenshu"] == pytest.approx(1234.0, abs=1.0)
    assert result["operating_profit"] == pytest.approx(123.0, abs=1.0)
    assert result["net_income"] == pytest.approx(45.0, abs=1.0)
    assert result["total_assets"] == pytest.approx(3000.0, abs=1.0)
    assert result["equity_ratio"] == pytest.approx(38.5, abs=0.1)
    assert result["edinet_code"] == "E12345"
    assert result["fiscal_year_retrieved"] == 2024
    assert result["error"] is None


# ---------------------------------------------------------------------------
# test_1102: AC-1102 — TTL内キャッシュ → source="cache", HTTP未発行
# ---------------------------------------------------------------------------

def test_1102_cache_hit(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    # キャッシュに1時間前のレコードを挿入
    conn = sqlite3.connect(db_path)
    fetched_at = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO edinet_cache
        (corporate_number, fiscal_year, edinet_code, nenshu, operating_profit,
         net_income, total_assets, equity_ratio, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        ("1234567890123", 2024, "E12345", 1234.0, 123.0, 45.0, 3000.0, 38.5, fetched_at),
    )
    conn.commit()
    conn.close()

    with patch("requests.get") as mock_get:
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, use_cache=True, db_path=db_path
        )
        mock_get.assert_not_called()

    assert result["source"] == "cache"
    assert result["success"] is True


# ---------------------------------------------------------------------------
# test_1103: AC-1103 — TTL超過（25時間前）→ source="edinet"（再取得）
# ---------------------------------------------------------------------------

def test_1103_cache_expired(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    conn = sqlite3.connect(db_path)
    fetched_at = (datetime.now() - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO edinet_cache
        (corporate_number, fiscal_year, edinet_code, nenshu, operating_profit,
         net_income, total_assets, equity_ratio, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        ("1234567890123", 2024, "E12345", 999.0, 99.0, 9.0, 999.0, 10.0, fetched_at),
    )
    conn.commit()
    conn.close()

    with patch("requests.get", side_effect=_mock_requests_get()):
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, use_cache=True, db_path=db_path
        )

    assert result["source"] == "edinet"


# ---------------------------------------------------------------------------
# test_1104: AC-1104 — 12桁法人番号 → success=False, source="fallback"
# ---------------------------------------------------------------------------

def test_1104_invalid_corporate_number(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    result = fetch_edinet_financials("123456789012", db_path=db_path)  # 12桁
    assert result["success"] is False
    assert result["source"] == "fallback"
    assert result["error"] is not None
    assert "invalid" in result["error"].lower()


# ---------------------------------------------------------------------------
# test_1105: AC-1105 — タイムアウトモック → 例外なし, fallback
# ---------------------------------------------------------------------------

def test_1105_timeout_fallback(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    with patch("requests.get", side_effect=_mock_requests_get(timeout=True)):
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, db_path=db_path
        )
    assert result["success"] is False
    assert result["source"] == "fallback"


# ---------------------------------------------------------------------------
# test_1106: AC-1106 — HTTP 500 → fallback
# ---------------------------------------------------------------------------

def test_1106_http_500_fallback(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    with patch("requests.get", side_effect=_mock_requests_get(http_error_status=500)):
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, db_path=db_path
        )
    assert result["success"] is False
    assert result["source"] == "fallback"


# ---------------------------------------------------------------------------
# test_1107: AC-1107 — 正常取得後にキャッシュが1件存在する
# ---------------------------------------------------------------------------

def test_1107_cache_written_after_fetch(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    with patch("requests.get", side_effect=_mock_requests_get()):
        fetch_edinet_financials("1234567890123", fiscal_year=2024, db_path=db_path)

    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM edinet_cache WHERE corporate_number=? AND fiscal_year=?",
        ("1234567890123", 2024),
    ).fetchone()[0]
    conn.close()
    assert count == 1


# ---------------------------------------------------------------------------
# test_1108: AC-1108 — use_cache=False → キャッシュ無視して API が呼ばれる
# ---------------------------------------------------------------------------

def test_1108_use_cache_false(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    # 有効なキャッシュを挿入
    conn = sqlite3.connect(db_path)
    fetched_at = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO edinet_cache
        (corporate_number, fiscal_year, edinet_code, nenshu, operating_profit,
         net_income, total_assets, equity_ratio, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        ("1234567890123", 2024, "E12345", 999.0, 99.0, 9.0, 999.0, 10.0, fetched_at),
    )
    conn.commit()
    conn.close()

    with patch("requests.get", side_effect=_mock_requests_get()) as mock_get:
        result = fetch_edinet_financials(
            "1234567890123", fiscal_year=2024, use_cache=False, db_path=db_path
        )
        assert mock_get.called

    assert result["source"] == "edinet"


# ---------------------------------------------------------------------------
# test_1109: AC-1109 — 不明法人番号 → None, 例外なし
# ---------------------------------------------------------------------------

def test_1109_unknown_corporate_number_resolve(tmp_path):
    # resolve_edinet_code が空のリストを返す
    empty_response = MagicMock()
    empty_response.raise_for_status.return_value = None
    empty_response.json.return_value = {"results": []}

    with patch("requests.get", return_value=empty_response):
        result = resolve_edinet_code("9999999999999")

    assert result is None


# ---------------------------------------------------------------------------
# test_1110: AC-1110 — キャッシュヒット10回 → 500ms 以内
# ---------------------------------------------------------------------------

def test_1110_cache_hit_performance(tmp_path):
    db_path = _make_in_memory_db_path(tmp_path)
    conn = sqlite3.connect(db_path)
    fetched_at = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO edinet_cache
        (corporate_number, fiscal_year, edinet_code, nenshu, operating_profit,
         net_income, total_assets, equity_ratio, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        ("1234567890123", 2024, "E12345", 1234.0, 123.0, 45.0, 3000.0, 38.5, fetched_at),
    )
    conn.commit()
    conn.close()

    with patch("requests.get") as mock_get:
        start = time.perf_counter()
        for _ in range(10):
            result = fetch_edinet_financials(
                "1234567890123", fiscal_year=2024, use_cache=True, db_path=db_path
            )
            assert result["source"] == "cache"
        elapsed_ms = (time.perf_counter() - start) * 1000
        mock_get.assert_not_called()

    assert elapsed_ms < 500, f"10回のキャッシュ取得に {elapsed_ms:.1f}ms かかった（上限 500ms）"


# ---------------------------------------------------------------------------
# 追加テスト: _validate_corporate_number の境界値
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corp_no,expected", [
    ("1234567890123", True),   # 正常13桁
    ("123456789012", False),   # 12桁
    ("12345678901234", False),  # 14桁
    ("123456789012a", False),  # 英字混在
    ("", False),
])
def test_validate_corporate_number(corp_no, expected):
    from edinet_collector import _validate_corporate_number
    assert _validate_corporate_number(corp_no) == expected


# ---------------------------------------------------------------------------
# 追加テスト: XBRL パースの基本動作
# ---------------------------------------------------------------------------

def test_parse_xbrl_financials_basic():
    xbrl_zip_bytes = _make_xbrl_zip(
        nenshu_yen=5_000_000_000,
        op_profit_yen=500_000_000,
        net_income_yen=200_000_000,
        assets_yen=10_000_000_000,
        equity_ratio_pct=40.0,
    )
    # zip から XBRL バイト列を抽出
    with zipfile.ZipFile(io.BytesIO(xbrl_zip_bytes)) as zf:
        xbrl_bytes = zf.read(zf.namelist()[0])

    result = _parse_xbrl_financials(xbrl_bytes)
    assert result["nenshu"] == pytest.approx(5000.0, abs=1.0)
    assert result["operating_profit"] == pytest.approx(500.0, abs=1.0)
    assert result["net_income"] == pytest.approx(200.0, abs=1.0)
    assert result["total_assets"] == pytest.approx(10000.0, abs=1.0)
    assert result["equity_ratio"] == pytest.approx(40.0, abs=0.1)


# ---------------------------------------------------------------------------
# 企業一覧キャッシュ（edinet_company_list）テスト
# ---------------------------------------------------------------------------

def _make_companies_list(corp_no: str = "1234567890123", code: str = "E12345") -> list[dict]:
    return [
        {"corporateNumber": corp_no, "edinetCode": code, "filerName": "テスト株式会社"},
        {"corporateNumber": "9999999999991", "edinetCode": "E99999", "filerName": "他社"},
    ]


def test_company_cache_empty_on_fresh_db(tmp_path):
    """新規DBは企業一覧キャッシュなし → _is_company_cache_fresh が False を返す。"""
    db_path = str(tmp_path / "test.db")
    conn = _get_db_connection(db_path)
    assert _is_company_cache_fresh(conn) is False
    conn.close()


def test_company_cache_write_and_lookup(tmp_path):
    """企業一覧を書き込んだ後に法人番号でルックアップできる。"""
    db_path = str(tmp_path / "test.db")
    conn = _get_db_connection(db_path)
    _write_company_list_cache(conn, _make_companies_list())
    assert _is_company_cache_fresh(conn) is True
    assert _lookup_edinet_code_from_cache(conn, "1234567890123") == "E12345"
    assert _lookup_edinet_code_from_cache(conn, "9999999999991") == "E99999"
    assert _lookup_edinet_code_from_cache(conn, "0000000000000") is None
    conn.close()


def test_resolve_edinet_code_uses_company_cache(tmp_path):
    """キャッシュが有効なら companies.json への HTTP リクエストが発行されない。"""
    db_path = str(tmp_path / "test.db")
    conn = _get_db_connection(db_path)
    _write_company_list_cache(conn, _make_companies_list("1234567890123", "E12345"))
    conn.close()

    with patch("requests.get") as mock_get:
        result = resolve_edinet_code("1234567890123", db_path=db_path)
        mock_get.assert_not_called()

    assert result == "E12345"


def test_resolve_edinet_code_refreshes_stale_cache(tmp_path):
    """企業一覧キャッシュが期限切れなら API を叩いてキャッシュを更新する。"""
    import sqlite3 as _sqlite3

    db_path = str(tmp_path / "test.db")
    conn = _get_db_connection(db_path)
    # 8日前のキャッシュを挿入
    from datetime import datetime, timedelta
    old_time = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO edinet_company_list (corporate_number, edinet_code, filer_name, fetched_at) VALUES (?,?,?,?)",
        ("1234567890123", "E_OLD", "旧社名", old_time),
    )
    conn.commit()
    conn.close()

    fresh_response = MagicMock()
    fresh_response.raise_for_status.return_value = None
    fresh_response.json.return_value = {
        "results": [{"corporateNumber": "1234567890123", "edinetCode": "E_NEW", "filerName": "新社名"}]
    }

    with patch("requests.get", return_value=fresh_response) as mock_get:
        result = resolve_edinet_code("1234567890123", db_path=db_path)
        mock_get.assert_called_once()

    assert result == "E_NEW"

    # キャッシュが更新されていること
    conn2 = _sqlite3.connect(db_path)
    row = conn2.execute(
        "SELECT edinet_code FROM edinet_company_list WHERE corporate_number=?",
        ("1234567890123",),
    ).fetchone()
    conn2.close()
    assert row[0] == "E_NEW"


def test_resolve_edinet_code_not_in_fresh_cache_returns_none(tmp_path):
    """キャッシュが有効でも対象法人番号が存在しない場合は None を返す（API は叩かない）。"""
    db_path = str(tmp_path / "test.db")
    conn = _get_db_connection(db_path)
    _write_company_list_cache(conn, _make_companies_list("9999999999991", "E99999"))
    conn.close()

    with patch("requests.get") as mock_get:
        result = resolve_edinet_code("1234567890123", db_path=db_path)
        mock_get.assert_not_called()

    assert result is None


def test_fetch_edinet_uses_company_cache_no_extra_api_call(tmp_path):
    """fetch_edinet_financials が企業一覧キャッシュを利用し、
    companies.json 呼び出しなしで正常取得できる。"""
    db_path = _make_in_memory_db_path(tmp_path)
    # 企業一覧キャッシュを事前に投入
    conn = _get_db_connection(db_path)
    _write_company_list_cache(conn, _make_companies_list("1234567890123", "E12345"))
    conn.close()

    call_log: list[str] = []

    def _side_effect(url, **kwargs):
        call_log.append(url)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "companies.json" in url:
            resp.json.return_value = {"results": []}  # 呼ばれても空を返す
        elif "documents.json" in url:
            resp.json.return_value = _make_documents_response("E12345", "S100XXXX")
        elif "documents/S100XXXX" in url:
            resp.content = _make_xbrl_zip()
        else:
            resp.json.return_value = {"results": []}
        return resp

    with patch("requests.get", side_effect=_side_effect):
        result = fetch_edinet_financials("1234567890123", fiscal_year=2024, db_path=db_path)

    assert result["success"] is True
    assert not any("companies.json" in u for u in call_log), \
        f"companies.json が呼ばれた: {call_log}"
