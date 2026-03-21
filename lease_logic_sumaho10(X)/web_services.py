"""
web_services.py — 外部データ取得・キャッシュ・検索・スクレイピング関連モジュール。

含む機能:
- 業界目安・トレンド・資産目安のネット検索とJSONキャッシュ管理
- 補助金・設備・リース分類のローカルデータ検索
- 記事スクレイピング・日本語判定
- 市場金利（JGB）読み込み
- AIアドバイス用コンテキスト生成
- 業種別統計（get_stats）
"""

import os
import json
import datetime
import re
from collections import Counter

import streamlit as st

from config import (
    BASE_DIR,
    WEB_BENCHMARKS_FILE,
    TRENDS_EXTENDED_FILE,
    ASSETS_BENCHMARKS_FILE,
    SALES_BAND_FILE,
)
from data_cases import load_all_cases


# ─── ローカルJSONデータの遅延ロード ────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_subsidy_schedule_data: dict | None = None
_useful_life_data: dict | None = None
_lease_classification_data: dict | None = None
_jgb_rates_data: dict | None = None


def _get_subsidy_schedule() -> dict:
    global _subsidy_schedule_data
    if _subsidy_schedule_data is None:
        _subsidy_schedule_data = _load_json("subsidy_schedule.json")
    return _subsidy_schedule_data


def _get_useful_life() -> dict:
    global _useful_life_data
    if _useful_life_data is None:
        _useful_life_data = _load_json("useful_life_equipment.json")
    return _useful_life_data


def _get_lease_classification() -> dict:
    global _lease_classification_data
    if _lease_classification_data is None:
        _lease_classification_data = _load_json("lease_classification.json")
    return _lease_classification_data


def _get_jgb_rates() -> dict:
    global _jgb_rates_data
    if _jgb_rates_data is None:
        _jgb_rates_data = _load_json("jgb_rates.json")
    return _jgb_rates_data


# ─── ネット業界目安キャッシュ ────────────────────────────────────────────────

_WEB_BENCH_KEYS = [
    "op_margin", "equity_ratio", "gross_margin", "ord_margin", "net_margin", "dep_ratio",
    "roa", "roe", "asset_turnover", "fixed_ratio", "debt_ratio",
    "fixed_to_equity", "debt_to_equity", "fixed_asset_turnover", "current_asset_ratio", "current_ratio",
]


def _get_benchmark_cutoff_date() -> datetime.date:
    """業界目安を『年1回・4月1日』で更新するための基準日。"""
    today = datetime.date.today()
    april1_this = datetime.date(today.year, 4, 1)
    if today >= april1_this:
        return april1_this
    return datetime.date(today.year - 1, 4, 1)


def _load_web_benchmarks_cache() -> dict:
    """保存済みのネット業界目安を読み込む。"""
    if not os.path.exists(WEB_BENCHMARKS_FILE):
        return {}
    try:
        with open(WEB_BENCHMARKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_web_benchmark(industry_sub: str, data: dict):
    """中分類ごとの業界目安をファイルに追記・上書きする。"""
    cache = _load_web_benchmarks_cache()
    entry = {"fetched_at": datetime.date.today().isoformat(), "snippets": data.get("snippets", [])}
    for k in _WEB_BENCH_KEYS:
        v = data.get(k)
        if v is not None:
            entry[k] = v
    cache[industry_sub] = entry
    try:
        with open(WEB_BENCHMARKS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_json_cache(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_cache(filepath: str, data: dict):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 業界トレンド拡充・資産目安・売上規模帯のキャッシュ（4月1日基準で再利用）
trends_extended_cache: dict = {}
assets_benchmarks_cache: dict = {}
sales_band_cache: dict = {}


def _ensure_web_caches_loaded():
    global trends_extended_cache, assets_benchmarks_cache, sales_band_cache
    if not trends_extended_cache and os.path.exists(TRENDS_EXTENDED_FILE):
        trends_extended_cache.update(_load_json_cache(TRENDS_EXTENDED_FILE))
    if not assets_benchmarks_cache and os.path.exists(ASSETS_BENCHMARKS_FILE):
        assets_benchmarks_cache.update(_load_json_cache(ASSETS_BENCHMARKS_FILE))
    if not sales_band_cache and os.path.exists(SALES_BAND_FILE):
        sales_band_cache.update(_load_json_cache(SALES_BAND_FILE))


# ─── ネット検索・フェッチ ─────────────────────────────────────────────────────

def fetch_industry_trend_extended(industry_sub: str, force_refresh: bool = False) -> str:
    """業界トレンドをネットで検索して拡充テキストを保存。4月1日基準でキャッシュ有効。"""
    if not industry_sub:
        return ""
    _ensure_web_caches_loaded()
    cutoff = _get_benchmark_cutoff_date()
    cached = trends_extended_cache.get(industry_sub)
    if cached and not force_refresh:
        try:
            if datetime.date.fromisoformat(cached.get("fetched_at", "")) >= cutoff:
                return cached.get("text", "") or ""
        except (ValueError, TypeError):
            pass
    query = f"{industry_sub} 業界動向 2025 課題 見通し"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return ""
    text_parts = []
    for r in results[:4]:
        body = (r.get("body") or "").strip()
        if body:
            text_parts.append(body[:400])
    text = "\n".join(text_parts)[:2000] if text_parts else ""
    trends_extended_cache[industry_sub] = {"fetched_at": datetime.date.today().isoformat(), "text": text}
    _save_json_cache(TRENDS_EXTENDED_FILE, trends_extended_cache)
    return text


def fetch_industry_assets_from_web(industry_sub: str, force_refresh: bool = False) -> dict:
    """業種別の総資産・流動比率の目安をネット検索して保存。"""
    _ensure_web_caches_loaded()
    out = {"total_assets_note": "", "current_ratio": None}
    if not industry_sub:
        return out
    cached = assets_benchmarks_cache.get(industry_sub)
    if cached and not force_refresh:
        try:
            if datetime.date.fromisoformat(cached.get("fetched_at", "")) >= _get_benchmark_cutoff_date():
                return {k: cached.get(k) for k in ["total_assets_note", "current_ratio"]}
        except (ValueError, TypeError):
            pass
    query = f"{industry_sub} 業界 総資産 流動比率 目安 平均"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return out
    combined = " ".join([(r.get("body") or "") for r in results])
    m = re.search(r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", combined)
    if m:
        try:
            out["current_ratio"] = float(m.group(1))
        except ValueError:
            pass
    out["total_assets_note"] = combined[:500] if combined else ""
    assets_benchmarks_cache[industry_sub] = {"fetched_at": datetime.date.today().isoformat(), **out}
    _save_json_cache(ASSETS_BENCHMARKS_FILE, assets_benchmarks_cache)
    return out


def fetch_sales_band_benchmarks(force_refresh: bool = False) -> str:
    """売上規模帯別の利益率等をネット検索して保存。全体で1件のキャッシュ。"""
    _ensure_web_caches_loaded()
    if sales_band_cache.get("fetched_at") and not force_refresh:
        try:
            if datetime.date.fromisoformat(sales_band_cache["fetched_at"]) >= _get_benchmark_cutoff_date():
                return sales_band_cache.get("text", "")
        except (ValueError, TypeError):
            pass
    query = "中小企業 売上規模 利益率 平均 売上高別 統計"
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, region="jp-jp", max_results=4))
    except Exception:
        return ""
    text = "\n".join([(r.get("body") or "")[:300] for r in results[:4]])
    sales_band_cache["fetched_at"] = datetime.date.today().isoformat()
    sales_band_cache["text"] = text
    _save_json_cache(SALES_BAND_FILE, sales_band_cache)
    return text


def _parse_benchmark_number(text: str, patterns: list) -> float | None:
    """テキストから正規表現で最初にマッチした数値を返す。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def fetch_industry_benchmarks_from_web(industry_sub: str, force_refresh: bool = False) -> dict:
    """
    中分類ごとにネット検索で業界目安を取得する。
    取得結果は web_industry_benchmarks.json に保存し、年1回（4月1日を境）にだけ再検索する。
    返却: {"snippets": [...], "op_margin": float or None, ...}
    """
    out = {k: None for k in _WEB_BENCH_KEYS}
    out["snippets"] = []
    if not industry_sub:
        return out
    if not force_refresh:
        cutoff = _get_benchmark_cutoff_date()
        cache = _load_web_benchmarks_cache()
        cached = cache.get(industry_sub)
        if cached:
            try:
                fetched = datetime.date.fromisoformat(cached["fetched_at"])
                if fetched >= cutoff:
                    ret = {"snippets": cached.get("snippets", [])}
                    for k in _WEB_BENCH_KEYS:
                        if k in cached and cached[k] is not None:
                            ret[k] = cached[k]
                    return ret
            except (ValueError, TypeError):
                pass
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        ddgs = DDGS()
    except Exception:
        _save_web_benchmark(industry_sub, out)
        return out

    combined_text = ""
    query1 = f"{industry_sub} 業界 営業利益率 自己資本比率 平均 業界動向"
    try:
        results1 = list(ddgs.text(query1, region="jp-jp", max_results=5))
    except Exception:
        results1 = []
    for r in results1:
        title, body, href = (r.get("title") or ""), (r.get("body") or ""), (r.get("href") or "")
        out["snippets"].append({"title": title, "body": body, "href": href})
        combined_text += title + " " + body + " "
    query2 = f"{industry_sub} 業界 売上高総利益率 経常利益率 ROA 流動比率 借入金 平均 目安"
    try:
        results2 = list(ddgs.text(query2, region="jp-jp", max_results=5))
    except Exception:
        results2 = []
    for r in results2:
        title, body = (r.get("title") or ""), (r.get("body") or "")
        out["snippets"].append({"title": title, "body": body, "href": r.get("href") or ""})
        combined_text += title + " " + body + " "

    def parse(patterns):
        return _parse_benchmark_number(combined_text, patterns)

    if out["op_margin"] is None:
        out["op_margin"] = parse([r"営業利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"営業利益[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["equity_ratio"] is None:
        out["equity_ratio"] = parse([r"自己資本比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["gross_margin"] is None:
        out["gross_margin"] = parse([r"売上高総利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"粗利率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"総利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["ord_margin"] is None:
        out["ord_margin"] = parse([r"経常利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"経常利益[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["net_margin"] is None:
        out["net_margin"] = parse([r"当期純利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"純利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["dep_ratio"] is None:
        out["dep_ratio"] = parse([r"減価償却費[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"減価償却[^/]*/?\s*売上[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["roa"] is None:
        out["roa"] = parse([r"ROA[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"総資産利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["roe"] is None:
        out["roe"] = parse([r"ROE[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"自己資本利益率[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["asset_turnover"] is None:
        out["asset_turnover"] = parse([r"総資産回転率[^\d]*([0-9]+\.?[0-9]*)\s*回?", r"総資産回転[^\d]*([0-9]+\.?[0-9]*)"])
    if out["fixed_ratio"] is None:
        out["fixed_ratio"] = parse([r"固定資産比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"固定資産[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["debt_ratio"] is None:
        out["debt_ratio"] = parse([r"借入金等依存度[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"借入金[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"有利子負債[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["fixed_to_equity"] is None:
        out["fixed_to_equity"] = parse([r"固定比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"固定資産[^\d]*/[^\d]*自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["debt_to_equity"] is None:
        out["debt_to_equity"] = parse([r"負債比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"負債[^\d]*/[^\d]*自己資本[^\d]*([0-9]+\.?[0-9]*)\s*%?"])
    if out["fixed_asset_turnover"] is None:
        out["fixed_asset_turnover"] = parse([r"固定資産回転率[^\d]*([0-9]+\.?[0-9]*)\s*回?", r"固定資産回転[^\d]*([0-9]+\.?[0-9]*)"])
    if out["current_asset_ratio"] is None:
        out["current_asset_ratio"] = parse([r"流動資産比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"流動資産[^\d]*([0-9]+\.?[0-9]*)\s*%"])
    if out["current_ratio"] is None:
        out["current_ratio"] = parse([r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%?", r"流動比率[^\d]*([0-9]+\.?[0-9]*)\s*%"])

    _save_web_benchmark(industry_sub, out)
    return out


# ─── キャッシュ読み取り（同期・UI向け） ──────────────────────────────────────

def get_trend_extended(industry_sub: str) -> str:
    """業界トレンド拡充テキストを返す（キャッシュがあればそれ、なければ空）。"""
    _ensure_web_caches_loaded()
    c = trends_extended_cache.get(industry_sub)
    return (c.get("text") or "") if c else ""


def get_assets_benchmark(industry_sub: str) -> dict:
    """業種別資産目安を返す。"""
    _ensure_web_caches_loaded()
    c = assets_benchmarks_cache.get(industry_sub)
    return c if c else {}


def get_sales_band_text() -> str:
    """売上規模帯別指標のテキストを返す。"""
    _ensure_web_caches_loaded()
    return sales_band_cache.get("text", "") or ""


def get_all_industry_sub_for_benchmarks() -> list:
    """業界目安を取得すべき中分類の一覧（重複なし）。"""
    benchmarks_data = _load_json("industry_benchmarks.json")
    subs = set()
    if benchmarks_data:
        subs.update(benchmarks_data.keys())
    for c in load_all_cases():
        sub = c.get("industry_sub")
        if sub:
            subs.add(sub)
    return sorted(subs)


# ─── ローカルデータ検索 ───────────────────────────────────────────────────────

def search_subsidies_by_industry(industry_sub: str) -> list:
    """業種に紐づく補助金一覧を返す。subsidy_schedule.json の industries で検索。"""
    data = _get_subsidy_schedule()
    out = []
    for s in (data.get("subsidies") or []):
        if industry_sub in (s.get("industries") or []):
            out.append(s)
    return out


def search_equipment_by_keyword(keyword: str) -> list:
    """耐用年数データからキーワードで設備を検索。"""
    data = _get_useful_life()
    if not keyword or not data:
        return []
    out = []
    kw = keyword.strip().lower()
    for cat in (data.get("categories") or []):
        for item in (cat.get("items") or []):
            name = (item.get("name") or "")
            if kw in name.lower():
                out.append({"category": cat.get("name"), **item})
    return out


def get_lease_classification_text() -> str:
    """リース判定フローと契約形態別条件の要約を返す。"""
    data = _get_lease_classification()
    if not data:
        return ""
    lines = ["【リース判定の目安】"]
    for step in (data.get("classification_flow") or []):
        lines.append(f"Step{step.get('step')}: {step.get('question')} → {step.get('yes_go') or step.get('no_go')}")
    lines.append("")
    for ct in (data.get("contract_types") or []):
        lines.append(f"■ {ct.get('type')}: {ct.get('summary')}")
        for t in (ct.get("typical_conditions") or [])[:3]:
            lines.append(f"  - {t}")
    return "\n".join(lines)


# ─── スクレイピング・テキスト判定 ────────────────────────────────────────────

def scrape_article_text(url: str):
    """指定されたURLから記事本文をスクレイピングする（簡易版）。"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        st.error("記事読み込み機能には追加ライブラリが必要です: pip install requests beautifulsoup4")
        return None

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.3"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        article_body = soup.find("article") or soup.find("main") or soup.body
        if article_body is None:
            return "本文を抽出できませんでした。ページ構造を解析できません。"

        paragraphs = article_body.find_all("p")
        if not paragraphs:
            return "本文を抽出できませんでした。本文らしき段落が見つかりません。"

        text = " ".join(p.get_text() for p in paragraphs)
        return text[:5000] if text else "本文を抽出できませんでした。"
    except Exception as e:
        return f"記事の読み込みに失敗しました: {e}"


def is_japanese_text(text: str, threshold: float = 0.2) -> bool:
    """テキスト中に日本語（ひらがな・カタカナ・漢字）が一定割合以上含まれるかを判定する。"""
    if not text:
        return False
    jp_count = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if (
            ("\u3040" <= ch <= "\u30ff")    # ひらがな・カタカナ
            or ("\u4e00" <= ch <= "\u9faf") # CJK統合漢字
            or ("\uff66" <= ch <= "\uff9d") # 半角カナ
        ):
            jp_count += 1
    if total == 0:
        return False
    return jp_count / total >= threshold


# ─── 市場金利 ─────────────────────────────────────────────────────────────────

def get_market_rate(year_month, term_years: int = 5) -> float:
    """JGB金利データから市場金利を返す。"""
    jgb_rates = _get_jgb_rates()
    if year_month not in jgb_rates:
        keys = sorted(jgb_rates.keys())
        if keys:
            year_month = keys[-1]
        else:
            return 1.0
    rate_data = jgb_rates[year_month]
    if term_years >= 8:
        return rate_data.get("10y", 1.0)
    else:
        return rate_data.get("5y", 0.5)


# ─── 倒産・最新動向の検索 ────────────────────────────────────────────────────

def search_bankruptcy_trends(industry_sub: str) -> str:
    """選択業界の最新の倒産トレンド・リスク情報を duckduckgo-search で検索する。"""
    try:
        from duckduckgo_search import DDGS
        query = f"{industry_sub} 業界 倒産 トレンド リスク 動向"
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region="jp-jp", max_results=5))
        if not results:
            return "（該当業界の倒産トレンド情報は取得できませんでした）"
        summary = ""
        for r in results:
            summary += f"- {r.get('title', '')}: {r.get('body', '')[:200]}…\n"
        return summary.strip()
    except Exception as e:
        return f"（業界リスク検索エラー: {e}）"


def search_latest_trends(query: str) -> str:
    """最新の業界動向をネットで検索してテキストで返す。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords=query, region="jp-jp", max_results=3))
            if not results:
                return "検索結果が見つかりませんでした。"
            summary = "\n【ネット検索による最新動向】\n"
            for res in results:
                summary += f"- {res['title']}: {res['body']} ({res['href']})\n"
            return summary
    except Exception as e:
        return f"\n（検索エラーにより最新情報の取得に失敗しました: {e}）"


# ─── 業種別統計 ───────────────────────────────────────────────────────────────

def get_stats(target_sub_industry: str) -> dict:
    """業種別の過去案件統計を返す。"""
    cases = load_all_cases()
    target_cases = [c for c in cases if c.get("industry_sub") == target_sub_industry]
    count = len(target_cases)

    if count == 0:
        return {
            "count": 0, "closed_count": 0, "avg_score": 0.0,
            "approved_count": 0, "close_rate": 0.0,
            "lost_reasons": [], "top_competitors_lost": [], "avg_winning_rate": None,
        }

    scores = [c["result"]["score"] for c in target_cases if "result" in c]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    approved_count = len([s for s in scores if s >= 70])

    closed_cases = [c for c in target_cases if c.get("final_status") == "成約"]
    lost_cases = [c for c in target_cases if c.get("final_status") == "失注"]
    total_finished = len(closed_cases) + len(lost_cases)
    close_rate = len(closed_cases) / total_finished if total_finished > 0 else 0.0

    lost_reasons = [c.get("lost_reason") for c in lost_cases if c.get("lost_reason")]
    competitor_names = [c.get("competitor_name", "").strip() for c in lost_cases if c.get("competitor_name")]
    top_competitors_lost = []
    if competitor_names:
        counted = Counter(competitor_names)
        top_competitors_lost = [name for name, _ in counted.most_common(10)]

    winning_rates = [
        c.get("final_rate") for c in closed_cases
        if c.get("final_rate") is not None
        and isinstance(c.get("final_rate"), (int, float))
        and c.get("final_rate") > 0
    ]
    avg_winning_rate = sum(winning_rates) / len(winning_rates) if winning_rates else None

    return {
        "count": count,
        "closed_count": len(closed_cases),
        "avg_score": avg_score,
        "approved_count": approved_count,
        "close_rate": close_rate,
        "lost_reasons": lost_reasons,
        "top_competitors_lost": top_competitors_lost,
        "avg_winning_rate": avg_winning_rate,
    }


# ─── AIアドバイス用コンテキスト ──────────────────────────────────────────────

def get_advice_context_extras(selected_sub: str, selected_major: str) -> str:
    """AIアドバイス用に、補助金・耐用年数・リース分類・業界トレンド・資産目安・売上規模帯のテキストをまとめて返す。"""
    parts = []
    subs = search_subsidies_by_industry(selected_sub)
    if subs:
        parts.append("【該当業種の補助金例】")
        for s in subs[:5]:
            line = f"- {s.get('name')}: {s.get('summary')} 申請目安: {s.get('application_period')}"
            if s.get("url"):
                line += f" 問い合わせ先: {s.get('url')}"
            parts.append(line)
    lc = get_lease_classification_text()
    if lc:
        parts.append("\n" + lc)
    trend_ex = get_trend_extended(selected_sub)
    if trend_ex:
        parts.append("\n【業界トレンド（拡充）】\n" + trend_ex[:1200])
    ab = get_assets_benchmark(selected_sub)
    if ab.get("current_ratio") is not None:
        parts.append(f"\n【業界の資産目安】流動比率目安: {ab['current_ratio']}%")
    if ab.get("total_assets_note"):
        parts.append("総資産・業界メモ: " + ab["total_assets_note"][:300])
    sb = get_sales_band_text()
    if sb:
        parts.append("\n【売上規模帯別の目安】\n" + sb[:600])
    stats = get_stats(selected_sub)
    if stats.get("top_competitors_lost"):
        parts.append("\n【過去に負けが多い競合】" + "、".join(stats["top_competitors_lost"][:5]))
    if stats.get("avg_winning_rate") is not None and stats["avg_winning_rate"] > 0:
        parts.append(f"\n【同業種の平均成約金利】{stats['avg_winning_rate']:.2f}%")
    if stats.get("top_competitors_lost") or (stats.get("avg_winning_rate") and stats["avg_winning_rate"] > 0):
        parts.append("\n上記の競合動向・成約金利を踏まえ、競合に勝つための対策も考慮してアドバイスしてください。")
    return "\n".join(parts) if parts else ""
