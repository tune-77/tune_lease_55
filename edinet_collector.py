"""
EDINET データ自動収集モジュール（P4-002）
法人番号をキーに EDINET API から有価証券報告書の財務データを取得する。
"""
from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import time
import zipfile
from datetime import datetime, timedelta
from typing import Optional, TypedDict

import requests
from lxml import etree

logger = logging.getLogger(__name__)

EDINET_BASE = "https://disclosure.edinet-api.go.jp/api/v2"
API_TIMEOUT = 10
RATE_SLEEP = 0.5
COMPANY_CACHE_TTL_DAYS = 7


class EdinetFinancialsResult(TypedDict):
    success: bool
    source: str                   # "edinet" | "cache" | "fallback"
    nenshu: Optional[float]       # 売上高（百万円）
    operating_profit: Optional[float]
    net_income: Optional[float]
    total_assets: Optional[float]
    equity_ratio: Optional[float]
    fiscal_year_retrieved: Optional[int]
    edinet_code: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _validate_corporate_number(corporate_number: str) -> bool:
    """BR-411: 法人番号は13桁の数字であること。"""
    return isinstance(corporate_number, str) and corporate_number.isdigit() and len(corporate_number) == 13


def _fallback_result(error: str | None = None) -> EdinetFinancialsResult:
    return {
        "success": False,
        "source": "fallback",
        "nenshu": None,
        "operating_profit": None,
        "net_income": None,
        "total_assets": None,
        "equity_ratio": None,
        "fiscal_year_retrieved": None,
        "edinet_code": None,
        "error": error,
    }


def _get_db_connection(db_path: str) -> sqlite3.Connection:
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edinet_company_list (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            corporate_number    TEXT    NOT NULL UNIQUE,
            edinet_code         TEXT    NOT NULL,
            filer_name          TEXT,
            fetched_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def _is_company_cache_fresh(conn: sqlite3.Connection) -> bool:
    """企業一覧キャッシュが COMPANY_CACHE_TTL_DAYS 以内に取得済みかチェック。"""
    try:
        cutoff = (datetime.now() - timedelta(days=COMPANY_CACHE_TTL_DAYS)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row = conn.execute(
            "SELECT COUNT(*) FROM edinet_company_list WHERE fetched_at > ?",
            (cutoff,),
        ).fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False


def _lookup_edinet_code_from_cache(
    conn: sqlite3.Connection, corporate_number: str
) -> str | None:
    """キャッシュ済み企業一覧から edinetCode を返す。"""
    try:
        row = conn.execute(
            "SELECT edinet_code FROM edinet_company_list WHERE corporate_number = ?",
            (corporate_number,),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _write_company_list_cache(
    conn: sqlite3.Connection, companies: list[dict]
) -> None:
    """企業一覧を DB に全件 INSERT OR REPLACE（TTL リセット）。"""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            """
            INSERT OR REPLACE INTO edinet_company_list
            (corporate_number, edinet_code, filer_name, fetched_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    c.get("corporateNumber"),
                    c.get("edinetCode"),
                    c.get("filerName"),
                    now,
                )
                for c in companies
                if c.get("corporateNumber") and c.get("edinetCode")
            ],
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[edinet_collector] company_list cache write error: {e}")


def _check_cache(
    conn: sqlite3.Connection, corporate_number: str, fiscal_year: int
) -> EdinetFinancialsResult | None:
    """BR-412: TTL 24時間のキャッシュ確認。"""
    try:
        cutoff = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        row = conn.execute(
            """
            SELECT nenshu, operating_profit, net_income, total_assets, equity_ratio,
                   edinet_code, fetched_at
            FROM edinet_cache
            WHERE corporate_number = ? AND fiscal_year = ? AND fetched_at > ?
            """,
            (corporate_number, fiscal_year, cutoff),
        ).fetchone()
        if row:
            return {
                "success": True,
                "source": "cache",
                "nenshu": row[0],
                "operating_profit": row[1],
                "net_income": row[2],
                "total_assets": row[3],
                "equity_ratio": row[4],
                "fiscal_year_retrieved": fiscal_year,
                "edinet_code": row[5],
                "error": None,
                "fetched_at": row[6],
            }
        return None
    except Exception as e:
        logger.warning(f"[edinet_collector] cache read error: {e}")
        return None


def _write_cache(
    conn: sqlite3.Connection,
    corporate_number: str,
    fiscal_year: int,
    edinet_code: str | None,
    financials: dict,
    raw_json: str | None = None,
) -> None:
    """BR-416: EDINET 取得成功時にキャッシュへ INSERT OR REPLACE。"""
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO edinet_cache
            (corporate_number, fiscal_year, edinet_code, nenshu, operating_profit,
             net_income, total_assets, equity_ratio, raw_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                corporate_number,
                fiscal_year,
                edinet_code,
                financials.get("nenshu"),
                financials.get("operating_profit"),
                financials.get("net_income"),
                financials.get("total_assets"),
                financials.get("equity_ratio"),
                raw_json,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[edinet_collector] cache write error: {e}")


# ---------------------------------------------------------------------------
# XBRL パース
# ---------------------------------------------------------------------------

_XBRL_SEARCH_MAP = {
    "nenshu": [
        "NetSales",
        "Revenue",
        "NetSalesSummaryOfBusinessResults",
    ],
    "operating_profit": [
        "OperatingIncome",
        "OperatingProfit",
    ],
    "net_income": [
        "ProfitLossAttributableToOwnersOfParent",
        "NetIncome",
        "ProfitLoss",
    ],
    "total_assets": [
        "Assets",
        "TotalAssets",
    ],
    "equity_ratio": [
        "EquityToAssetRatio",
    ],
}

_EQUITY_TAGS = ["NetAssets", "Equity", "EquityAttributableToOwnersOfParent"]
_NS_PREFIXES = ["jppfs_cor", "jpigp_cor", "jpdei_cor"]


def _parse_xbrl_financials(xbrl_bytes: bytes) -> dict:
    """XBRL バイト列から財務数値を抽出する（百万円単位）。"""
    result: dict = {k: None for k in _XBRL_SEARCH_MAP}
    try:
        root = etree.fromstring(xbrl_bytes)
    except Exception as e:
        logger.warning(f"[edinet_collector] XBRL parse error: {e}")
        return result

    nsmap = {k: v for k, v in root.nsmap.items() if k is not None}

    def _find_value(tag_names: list[str], is_ratio: bool = False) -> Optional[float]:
        divisor = 1.0 if is_ratio else 1_000_000.0
        for prefix in _NS_PREFIXES:
            ns_uri = nsmap.get(prefix)
            if not ns_uri:
                continue
            for tag in tag_names:
                # BR-414: 連結優先
                for elem in root.iter(f"{{{ns_uri}}}{tag}"):
                    ctx = elem.get("contextRef", "")
                    if "Consoli" in ctx or "consoli" in ctx:
                        try:
                            return round(float(elem.text) / divisor, 1)
                        except (ValueError, TypeError):
                            pass
                for elem in root.iter(f"{{{ns_uri}}}{tag}"):
                    try:
                        return round(float(elem.text) / divisor, 1)
                    except (ValueError, TypeError):
                        pass
        return None

    for key, tag_names in _XBRL_SEARCH_MAP.items():
        result[key] = _find_value(tag_names, is_ratio=(key == "equity_ratio"))

    # 自己資本比率が取得できない場合は純資産/総資産で計算
    if result["equity_ratio"] is None and result["total_assets"] and result["total_assets"] > 0:
        equity = _find_value(_EQUITY_TAGS)
        if equity is not None:
            result["equity_ratio"] = round(equity / result["total_assets"] * 100, 1)

    return result


# ---------------------------------------------------------------------------
# EDINET API 呼び出し
# ---------------------------------------------------------------------------

def resolve_edinet_code(
    corporate_number: str,
    api_key: str | None = None,
    db_path: str | None = None,
) -> str | None:
    """法人番号から EDINET 企業コードを解決する。見つからない場合は None。

    db_path が指定された場合は edinet_company_list テーブルを 7 日 TTL で利用し、
    companies.json への不要なリクエストを抑制する。
    """
    conn: sqlite3.Connection | None = None
    if db_path:
        try:
            conn = _get_db_connection(db_path)
        except Exception as e:
            logger.warning(f"[edinet_collector] resolve_edinet_code db error: {e}")

    try:
        # キャッシュが有効なら DB ルックアップのみで解決
        if conn and _is_company_cache_fresh(conn):
            edinet_code = _lookup_edinet_code_from_cache(conn, corporate_number)
            if edinet_code is not None:
                logger.debug(
                    f"[edinet_collector] company_list cache hit for {corporate_number}"
                )
                return edinet_code
            # キャッシュに見つからない場合も API は叩かない（登録なし）
            logger.debug(
                f"[edinet_collector] company_list cache miss (not registered): {corporate_number}"
            )
            return None

        # キャッシュ未作成 or 期限切れ → API から全件取得してキャッシュ更新
        params: dict = {"type": "2"}
        if api_key:
            params["Subscription-Key"] = api_key
        time.sleep(RATE_SLEEP)
        resp = requests.get(
            f"{EDINET_BASE}/companies.json",
            params=params,
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        companies = data.get("results", [])

        if conn:
            _write_company_list_cache(conn, companies)

        for company in companies:
            if company.get("corporateNumber") == corporate_number:
                return company.get("edinetCode")
        return None
    except Exception as e:
        logger.warning(f"[edinet_collector] resolve_edinet_code error: {e}")
        return None
    finally:
        if conn:
            conn.close()


def _find_doc_id(edinet_code: str, fiscal_year: int, api_key: str | None) -> str | None:
    """有価証券報告書の docID を取得する（直近 90 日を逆順に検索）。"""
    end_date = datetime(fiscal_year + 1, 9, 30)
    start_date = datetime(fiscal_year, 1, 1)
    current = end_date

    while current >= start_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            params: dict = {"date": date_str, "type": "2"}
            if api_key:
                params["Subscription-Key"] = api_key
            time.sleep(RATE_SLEEP)
            resp = requests.get(
                f"{EDINET_BASE}/documents.json",
                params=params,
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            for doc in data.get("results", []):
                if (
                    doc.get("edinetCode") == edinet_code
                    and doc.get("formCode", "").startswith("030000")
                ):
                    return doc.get("docID")
        except Exception as e:
            logger.warning(f"[edinet_collector] _find_doc_id error on {date_str}: {e}")
        current -= timedelta(days=1)

    return None


def _download_and_parse_xbrl(doc_id: str, api_key: str | None) -> dict | None:
    """XBRL zip をダウンロードして財務数値を抽出する。"""
    try:
        params: dict = {"type": "5"}
        if api_key:
            params["Subscription-Key"] = api_key
        time.sleep(RATE_SLEEP)
        resp = requests.get(
            f"{EDINET_BASE}/documents/{doc_id}",
            params=params,
            timeout=API_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xbrl_names = [n for n in zf.namelist() if n.endswith(".xbrl")]
            fr_names = [n for n in xbrl_names if "_fr" in n or "jpcrp" in n.lower()]
            target = fr_names[0] if fr_names else (xbrl_names[0] if xbrl_names else None)
            if target is None:
                return None
            xbrl_bytes = zf.read(target)

        return _parse_xbrl_financials(xbrl_bytes)
    except Exception as e:
        logger.warning(f"[edinet_collector] _download_and_parse_xbrl error: {e}")
        return None


# ---------------------------------------------------------------------------
# メイン公開 API
# ---------------------------------------------------------------------------

def fetch_edinet_financials(
    corporate_number: str,
    fiscal_year: int | None = None,
    use_cache: bool = True,
    api_key: str | None = None,
    db_path: str = "data/lease_data.db",
) -> EdinetFinancialsResult:
    """
    法人番号をキーに EDINET API から財務データを取得する。

    取得できない場合は source="fallback" で全財務項目 None を返す。
    例外を外部に伝播させない。
    """
    # BR-411: フォーマット検証
    if not _validate_corporate_number(corporate_number):
        return _fallback_result(error="invalid corporate_number format")

    # API キー解決（環境変数優先）
    if api_key is None:
        api_key = os.getenv("EDINET_API_KEY")

    # 事業年度の決定（省略時は直近完了年度）
    if fiscal_year is None:
        today = datetime.now()
        fiscal_year = today.year - 1 if today.month <= 9 else today.year

    # DB 接続
    conn: sqlite3.Connection | None = None
    try:
        conn = _get_db_connection(db_path)
    except Exception as e:
        logger.warning(f"[edinet_collector] db connect error: {e}")

    # BR-412: キャッシュ確認
    if use_cache and conn is not None:
        cached = _check_cache(conn, corporate_number, fiscal_year)
        if cached is not None:
            logger.info(
                f"[edinet_collector] cache hit corporate_number={corporate_number} "
                f"fiscal_year={fiscal_year}"
            )
            conn.close()
            return cached

    # EDINET Code 解決（企業一覧キャッシュを利用）
    # conn は resolve_edinet_code が db_path を内部で開閉するため一旦クローズする
    if conn:
        conn.close()
        conn = None

    edinet_code: str | None = None
    try:
        edinet_code = resolve_edinet_code(corporate_number, api_key, db_path=db_path)
    except Exception as e:
        logger.warning(f"[edinet_collector] FALLBACK: resolve_edinet_code exception: {e}")

    if not edinet_code:
        logger.warning(
            f"[edinet_collector] FALLBACK: edinetCode not resolved for {corporate_number}"
        )
        return _fallback_result(error="edinetCode not found for the given corporate_number")

    # 書類 ID 取得
    doc_id: str | None = None
    try:
        doc_id = _find_doc_id(edinet_code, fiscal_year, api_key)
    except Exception as e:
        logger.warning(f"[edinet_collector] FALLBACK: _find_doc_id exception: {e}")

    if not doc_id:
        logger.warning(
            f"[edinet_collector] FALLBACK: annual report not found for "
            f"{edinet_code} {fiscal_year}"
        )
        return _fallback_result(error="annual report not found")

    # XBRL ダウンロード & パース
    financials: dict | None = None
    try:
        financials = _download_and_parse_xbrl(doc_id, api_key)
    except Exception as e:
        logger.warning(f"[edinet_collector] FALLBACK: XBRL parse exception: {e}")

    if not financials:
        logger.warning(f"[edinet_collector] FALLBACK: XBRL parse failed for {doc_id}")
        return _fallback_result(error="XBRL parse failed")

    logger.info(
        f"[edinet_collector] fetched corporate_number={corporate_number} "
        f"fiscal_year={fiscal_year}"
    )

    # BR-416: キャッシュ書き込み（DB 再接続）
    try:
        conn = _get_db_connection(db_path)
        _write_cache(conn, corporate_number, fiscal_year, edinet_code, financials)
        conn.close()
    except Exception as e:
        logger.warning(f"[edinet_collector] cache write error after fetch: {e}")

    return {
        "success": True,
        "source": "edinet",
        "nenshu": financials.get("nenshu"),
        "operating_profit": financials.get("operating_profit"),
        "net_income": financials.get("net_income"),
        "total_assets": financials.get("total_assets"),
        "equity_ratio": financials.get("equity_ratio"),
        "fiscal_year_retrieved": fiscal_year,
        "edinet_code": edinet_code,
        "error": None,
    }
