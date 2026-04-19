#!/usr/bin/env python3
"""
EDINET APIから財務データを取得して edinet_companies テーブルに保存。
対象: 製造業・卸売業・建設業の中小〜中堅企業（有価証券報告書）
最大 MAX_COMPANIES 社まで取得。
"""

import csv
import io
import json
import logging
import sqlite3
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

EDINET_BASE = "https://disclosure.edinet-api.go.jp/api/v2"
DB_PATH = Path(__file__).parent.parent / "data" / "screening_db.sqlite"
MAX_COMPANIES = 100
SLEEP_SEC = 0.5

# TSE 33業種コード: 建設業(1800,1810) / 製造業(3050〜3800) / 卸売業(6050,6100)
TARGET_INDUSTRY_PREFIXES = ("18", "30", "31", "32", "33", "34", "35", "36", "37", "38", "60", "61")
INDUSTRY_LABEL = {
    "18": "建設業",
    "30": "製造業", "31": "製造業", "32": "製造業", "33": "製造業",
    "34": "製造業", "35": "製造業", "36": "製造業", "37": "製造業",
    "38": "製造業", "60": "卸売業", "61": "卸売業",
}

# XBRL要素ID (namespace prefixは無視してsuffix比較)
XBRL_WANT = {
    "NetSales":               "revenue",
    "OperatingIncome":        "op_profit",
    "OrdinaryIncome":         "ordinary_income",
    "Assets":                 "total_assets",
    "NetAssets":              "net_assets",
    "CurrentAssets":          "current_assets",
    "CurrentLiabilities":     "current_liabilities",
    "NumberOfEmployees":      "employees",
    "InterestBearingDebt":    "interest_bearing_debt",
    "LongTermBorrowings":     "long_term_debt",
    "ShortTermBorrowingsAndCurrentPortionOfLongTermBorrowings": "short_term_debt",
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edinet_companies (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at          TEXT    NOT NULL,
            edinet_code         TEXT,
            company_name        TEXT,
            industry_code       TEXT,
            industry_name       TEXT,
            period_end          TEXT,
            revenue             INTEGER,
            op_profit           INTEGER,
            ordinary_income     INTEGER,
            total_assets        INTEGER,
            net_assets          INTEGER,
            current_assets      INTEGER,
            current_liabilities INTEGER,
            employees           INTEGER,
            interest_bearing_debt INTEGER,
            op_profit_ratio     REAL,
            equity_ratio        REAL,
            current_ratio       REAL,
            raw_json            TEXT
        )
    """)
    conn.commit()


def get_document_list(target_date: date) -> list[dict]:
    url = f"{EDINET_BASE}/documents.json"
    params = {"date": target_date.strftime("%Y-%m-%d"), "type": 2}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        log.warning("書類一覧取得失敗 %s: %s", target_date, e)
        return []


def parse_xbrl_csv(zip_bytes: bytes) -> dict:
    """XBRLのZIPからCSVを読んで財務指標を抽出。値は円単位(int)で返す。"""
    metrics: dict = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            for fname in csv_files:
                raw = zf.read(fname)
                # UTF-8 → CP932 フォールバック
                for enc in ("utf-8-sig", "cp932", "utf-8"):
                    try:
                        text = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    continue

                reader = csv.reader(io.StringIO(text))
                headers = None
                for row in reader:
                    if headers is None:
                        headers = row
                        continue
                    if len(row) < 2:
                        continue
                    # 要素IDが最初の列にある場合
                    element_id = row[0]
                    # namespace:ElementName 形式からsuffixを取得
                    suffix = element_id.split(":")[-1] if ":" in element_id else element_id
                    if suffix in XBRL_WANT:
                        key = XBRL_WANT[suffix]
                        if key in metrics:
                            continue  # 最初の値を採用
                        # 値の列を探す (最後の数値列)
                        value_str = ""
                        if headers:
                            for col_name in ("値", "value", "Value"):
                                if col_name in headers:
                                    idx = headers.index(col_name)
                                    if idx < len(row):
                                        value_str = row[idx]
                                        break
                        if not value_str:
                            # 最後の非空列を候補にする
                            for cell in reversed(row[1:]):
                                cell = cell.strip().replace(",", "")
                                if cell.lstrip("-").isdigit():
                                    value_str = cell
                                    break
                        if value_str:
                            value_str = value_str.strip().replace(",", "")
                            try:
                                metrics[key] = int(float(value_str))
                            except ValueError:
                                pass
    except Exception as e:
        log.debug("XBRL CSV パース失敗: %s", e)
    return metrics


def download_xbrl(doc_id: str) -> bytes | None:
    url = f"{EDINET_BASE}/documents/{doc_id}"
    params = {"type": 5}
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception as e:
        log.debug("XBRL DL失敗 %s: %s", doc_id, e)
    return None


def compute_derived(m: dict) -> dict:
    """派生指標(営業利益率・自己資本比率・流動比率)を計算。"""
    m = dict(m)
    rev = m.get("revenue")
    op = m.get("op_profit")
    assets = m.get("total_assets")
    net_assets = m.get("net_assets")
    cur_a = m.get("current_assets")
    cur_l = m.get("current_liabilities")

    m["op_profit_ratio"] = round(op / rev * 100, 2) if rev and op is not None and rev != 0 else None
    m["equity_ratio"] = round(net_assets / assets * 100, 2) if assets and net_assets is not None and assets != 0 else None
    m["current_ratio"] = round(cur_a / cur_l * 100, 2) if cur_l and cur_a is not None and cur_l != 0 else None
    return m


def save_company(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute("""
        INSERT INTO edinet_companies (
            fetched_at, edinet_code, company_name, industry_code, industry_name,
            period_end, revenue, op_profit, ordinary_income, total_assets, net_assets,
            current_assets, current_liabilities, employees, interest_bearing_debt,
            op_profit_ratio, equity_ratio, current_ratio, raw_json
        ) VALUES (
            datetime('now','localtime'),
            :edinet_code, :company_name, :industry_code, :industry_name,
            :period_end, :revenue, :op_profit, :ordinary_income, :total_assets, :net_assets,
            :current_assets, :current_liabilities, :employees, :interest_bearing_debt,
            :op_profit_ratio, :equity_ratio, :current_ratio, :raw_json
        )
    """, record)
    conn.commit()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # 直近90日分の日付を収集
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(1, 91)]

    saved = 0
    errors = 0
    seen_edinet_codes: set[str] = set()

    log.info("EDINET データ取得開始 (最大%d社)", MAX_COMPANIES)

    for target_date in dates:
        if saved >= MAX_COMPANIES:
            break

        docs = get_document_list(target_date)
        time.sleep(SLEEP_SEC)

        # 有価証券報告書 (docTypeCode=120) のみ
        yuho_docs = [d for d in docs if d.get("docTypeCode") == "120"]

        for doc in yuho_docs:
            if saved >= MAX_COMPANIES:
                break

            edinet_code = doc.get("edinetCode", "")
            if edinet_code in seen_edinet_codes:
                continue

            industry_code = str(doc.get("industryCode", ""))
            # 製造業・卸売業・建設業フィルタ
            matched_prefix = next(
                (p for p in TARGET_INDUSTRY_PREFIXES if industry_code.startswith(p)),
                None,
            )
            if not matched_prefix:
                continue

            industry_name = INDUSTRY_LABEL.get(matched_prefix, "その他")
            company_name = doc.get("filerName", "")
            doc_id = doc.get("docID", "")
            period_end = doc.get("periodEnd", "")

            log.info("[%d] %s (%s) %s doc_id=%s", saved + 1, company_name, industry_name, period_end, doc_id)

            xbrl_bytes = download_xbrl(doc_id)
            time.sleep(SLEEP_SEC)

            metrics: dict = {}
            if xbrl_bytes:
                metrics = parse_xbrl_csv(xbrl_bytes)
            else:
                log.debug("XBRL未取得: %s", doc_id)
                errors += 1

            metrics = compute_derived(metrics)

            record = {
                "edinet_code":          edinet_code,
                "company_name":         company_name,
                "industry_code":        industry_code,
                "industry_name":        industry_name,
                "period_end":           period_end,
                "revenue":              metrics.get("revenue"),
                "op_profit":            metrics.get("op_profit"),
                "ordinary_income":      metrics.get("ordinary_income"),
                "total_assets":         metrics.get("total_assets"),
                "net_assets":           metrics.get("net_assets"),
                "current_assets":       metrics.get("current_assets"),
                "current_liabilities":  metrics.get("current_liabilities"),
                "employees":            metrics.get("employees"),
                "interest_bearing_debt": metrics.get("interest_bearing_debt"),
                "op_profit_ratio":      metrics.get("op_profit_ratio"),
                "equity_ratio":         metrics.get("equity_ratio"),
                "current_ratio":        metrics.get("current_ratio"),
                "raw_json":             json.dumps(doc, ensure_ascii=False),
            }

            try:
                save_company(conn, record)
                seen_edinet_codes.add(edinet_code)
                saved += 1
            except Exception as e:
                log.error("DB保存失敗 %s: %s", company_name, e)
                errors += 1

    conn.close()
    log.info("完了: 取得件数=%d, エラー=%d", saved, errors)


if __name__ == "__main__":
    main()
