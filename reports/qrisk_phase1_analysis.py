#!/usr/bin/env python3
"""Q_risk フェーズ1 バッチ分析 — 成約外因子発見装置（試験実行）.

過去案件を3コホートに分割し、信用スコアで説明できない成約/失注差を探索する。
出力: reports/qrisk_phase1_discovery.md （DB配下には書き込まない）
"""
import sqlite3
import json
import statistics as st
from collections import Counter, defaultdict

DB = "data/lease_data.db"
OUT = "reports/qrisk_phase1_discovery.md"

WON = {"成約", "検収完了"}      # 実質成約
LOST = {"失注"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM past_cases").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = {}
        try:
            d = json.loads(r["data"]) if r["data"] else {}
        except Exception:
            d = {}
        inp = d.get("inputs", {}) or {}
        res = d.get("result", {}) or {}
        nenshu = f(inp.get("nenshu"))
        op = f(inp.get("op_profit"))
        rent = f(inp.get("rent"))
        bank = f(inp.get("bank_credit"))
        lease = f(inp.get("lease_credit"))
        rec = {
            "id": r["id"],
            "score": f(r["score"]),
            "user_eq": f(r["user_eq"]),
            "status": r["final_status"],
            "industry": r["industry_sub"],
            "dept": r["sales_dept"],
            "ts": r["timestamp"],
            "nenshu": nenshu,
            "op_profit": op,
            "op_margin": (op / nenshu * 100) if (op is not None and nenshu) else f(res.get("user_op")),
            "rent": rent,
            "rent_to_sales": (rent / nenshu * 100) if (rent is not None and nenshu) else None,
            "debt_to_sales": ((((bank or 0) + (lease or 0))) / nenshu * 100) if nenshu else None,
            "lease_term": f(inp.get("lease_term")),
            "acquisition_cost": f(inp.get("acquisition_cost")),
            "asset_score": f(inp.get("lease_asset_score")),
            "main_bank": d.get("main_bank"),
            "deal_source": inp.get("deal_source"),
            "customer_type": d.get("customer_type"),
            "competitor": d.get("competitor"),
            "competitor_rate": f(d.get("competitor_rate")),
            "final_rate": f(d.get("final_rate")),
            "base_rate_at_time": f(d.get("base_rate_at_time")),
            "winning_spread": f(d.get("winning_spread")),
            "lost_reason": (d.get("lost_reason") or "").strip(),
            "q_onehot": (((d.get("qualitative") or {}).get("onehot")) or {}),
            "contract_prob": f(res.get("contract_prob")),
        }
        out.append(rec)
    return out


def desc(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": st.mean(vals),
        "median": st.median(vals),
    }


def fmt(s, key):
    d = s.get(key) if isinstance(s, dict) else None
    return ""  # placeholder


def cohort_stats(rows):
    out = {}
    out["n"] = len(rows)
    out["won"] = sum(1 for r in rows if r["status"] in WON)
    out["lost"] = sum(1 for r in rows if r["status"] in LOST)
    for key in ["score", "op_margin", "rent_to_sales", "debt_to_sales",
                "nenshu", "lease_term", "acquisition_cost", "asset_score",
                "competitor_rate", "final_rate", "winning_spread"]:
        out[key] = desc([r[key] for r in rows])
    out["industry"] = Counter(r["industry"] for r in rows).most_common(8)
    out["dept"] = Counter(r["dept"] for r in rows).most_common(8)
    out["deal_source"] = Counter(r["deal_source"] for r in rows if r["deal_source"]).most_common(8)
    out["main_bank"] = Counter(r["main_bank"] for r in rows if r["main_bank"]).most_common(8)
    out["customer_type"] = Counter(r["customer_type"] for r in rows if r["customer_type"]).most_common(8)
    # 月別
    monthly = Counter()
    for r in rows:
        if r["ts"]:
            monthly[r["ts"][:7]] += 1
    out["monthly"] = sorted(monthly.items())
    return out


def main():
    rows = load_rows()
    total = len(rows)

    # コホート定義
    high_score_lost = [r for r in rows if r["score"] is not None and r["score"] >= 60 and r["status"] in LOST]
    low_score_won = [r for r in rows if r["score"] is not None and r["score"] < 40 and r["status"] in WON]
    same_band = [r for r in rows if r["score"] is not None and 40 <= r["score"] <= 60]
    same_won = [r for r in same_band if r["status"] in WON]
    same_lost = [r for r in same_band if r["status"] in LOST]

    L = []
    p = L.append
    p("# Q_risk フェーズ1 試験実行レポート — 成約外因子発見装置")
    p("")
    p("> 過去案件を3コホートに分割し、信用スコアでは説明できない成約/失注差を探索する。")
    p(f"> 実行日: 2026-06-03 / DB: `{DB}` / 対象: past_cases {total}件")
    p("")
    p("## 0. 母集団サマリ")
    p("")
    statuses = Counter(r["status"] for r in rows)
    p("| final_status | 件数 |")
    p("|---|---|")
    for s, c in statuses.most_common():
        p(f"| {s} | {c} |")
    p("")
    p("- 本分析では `成約`+`検収完了` を **実質成約(WON)**、`失注` を **失注(LOST)** とみなす。")
    sc = desc([r["score"] for r in rows])
    p(f"- スコア分布(全体): n={sc['n']}, min={sc['min']:.1f}, max={sc['max']:.1f}, mean={sc['mean']:.1f}, median={sc['median']:.1f}")
    p("")

    cohorts = [
        ("A. high_score_lost（スコア≥60なのに失注）", high_score_lost),
        ("B. low_score_won（スコア<40なのに実質成約）", low_score_won),
        ("C-won. same_score_split 成約側（スコア40-60）", same_won),
        ("C-lost. same_score_split 失注側（スコア40-60）", same_lost),
    ]

    def block(title, rs):
        s = cohort_stats(rs)
        p(f"## {title}")
        p("")
        p(f"- 件数: **{s['n']}** （実質成約 {s['won']} / 失注 {s['lost']}）")

        def line(label, key, unit=""):
            d = s.get(key)
            if d:
                p(f"- {label}: mean={d['mean']:.2f}{unit}, median={d['median']:.2f}{unit}, "
                  f"min={d['min']:.2f}, max={d['max']:.2f} (n={d['n']})")
            else:
                p(f"- {label}: データなし")
        line("スコア", "score")
        line("営業利益率", "op_margin", "%")
        line("リース料/売上", "rent_to_sales", "%")
        line("総借入/売上", "debt_to_sales", "%")
        line("売上高(千円)", "nenshu")
        line("リース期間(月)", "lease_term")
        line("取得価額", "acquisition_cost")
        line("物件スコア(換金性)", "asset_score")
        line("競合金利", "competitor_rate")
        line("最終金利", "final_rate")
        line("勝spread", "winning_spread")
        p(f"- 業種分布: {s['industry']}")
        p(f"- 営業部分布: {s['dept']}")
        p(f"- 案件ソース: {s['deal_source']}")
        p(f"- メイン銀行関係: {s['main_bank']}")
        p(f"- 顧客タイプ: {s['customer_type']}")
        p(f"- 月別件数: {s['monthly']}")
        p("")
        return s

    sA = block(cohorts[0][0], cohorts[0][1])
    sB = block(cohorts[1][0], cohorts[1][1])
    sCw = block(cohorts[2][0], cohorts[2][1])
    sCl = block(cohorts[3][0], cohorts[3][1])

    # ---- 発見タグ検出 ----
    p("## 発見タグ候補の検出")
    p("")
    tags = []

    # price_competition_gap: 高スコア失注の lost_reason / 金利
    hsl = high_score_lost
    price_reasons = sum(1 for r in hsl if any(k in (r["lost_reason"] or "") for k in ["金利", "価格", "条件", "他社", "競合"]))
    has_comp = sum(1 for r in hsl if r["competitor"] and r["competitor"] not in ("競合なし", "", None))
    if hsl:
        p(f"### price_competition_gap")
        p(f"- 高スコア失注 {len(hsl)}件中、lost_reason に金利/価格/条件/競合語を含む: **{price_reasons}件** ({price_reasons/len(hsl)*100:.0f}%)")
        p(f"- 競合あり案件: {has_comp}件 ({has_comp/len(hsl)*100:.0f}%)")
        reasons = Counter((r["lost_reason"] or "（空）")[:30] for r in hsl).most_common(10)
        p(f"- lost_reason 上位: {reasons}")
        if price_reasons / max(len(hsl), 1) >= 0.3:
            tags.append("price_competition_gap")
        p("")

    # bank_support_bridge: 低スコア成約の銀行支援
    lsw = low_score_won
    if lsw:
        bank_main = sum(1 for r in lsw if r["main_bank"] == "メイン先")
        bank_src = sum(1 for r in lsw if r["deal_source"] and "銀行" in r["deal_source"])
        bank_tag = sum(1 for r in lsw if r["q_onehot"].get("取引行と付き合い長い") or r["q_onehot"].get("既存返済懸念ない"))
        # 全体比較
        all_main = sum(1 for r in rows if r["main_bank"] == "メイン先") / max(total, 1)
        p(f"### bank_support_bridge")
        p(f"- 低スコア成約 {len(lsw)}件中: メイン先={bank_main} ({bank_main/len(lsw)*100:.0f}%), "
          f"銀行紹介ソース={bank_src} ({bank_src/len(lsw)*100:.0f}%), 銀行関連定性タグ={bank_tag}")
        p(f"- 参考: 全体のメイン先比率 {all_main*100:.0f}%")
        if (bank_main / len(lsw)) > all_main * 1.1 or bank_src / len(lsw) >= 0.3:
            tags.append("bank_support_bridge")
        p("")

    # asset_resale_anchor: 低スコア成約の物件スコア
    if lsw:
        asw = desc([r["asset_score"] for r in lsw])
        asall = desc([r["asset_score"] for r in rows])
        p(f"### asset_resale_anchor")
        if asw and asall:
            p(f"- 低スコア成約の物件スコア mean={asw['mean']:.1f} vs 全体 mean={asall['mean']:.1f}")
            if asw["mean"] > asall["mean"] + 3:
                tags.append("asset_resale_anchor")
        else:
            p("- 物件スコアデータ不足")
        p("")

    # sales_route_strength: 同スコア帯の営業部偏り（成約 vs 失注）
    if same_won and same_lost:
        wdept = Counter(r["dept"] for r in same_won)
        ldept = Counter(r["dept"] for r in same_lost)
        depts = set(wdept) | set(ldept)
        skew = []
        for d in depts:
            w, l = wdept.get(d, 0), ldept.get(d, 0)
            if w + l >= 10:
                rate = w / (w + l)
                skew.append((d, w, l, rate))
        skew.sort(key=lambda x: -x[3])
        base_rate = len(same_won) / (len(same_won) + len(same_lost))
        p(f"### sales_route_strength")
        p(f"- 同スコア帯(40-60)の基準成約率: {base_rate*100:.0f}%")
        p(f"- 営業部別成約率(件数10以上): ")
        for d, w, l, rate in skew[:10]:
            mark = " ★高" if rate >= base_rate + 0.15 else (" ▼低" if rate <= base_rate - 0.15 else "")
            p(f"  - {d}: 成約{w}/失注{l} = {rate*100:.0f}%{mark}")
        if any(abs(rate - base_rate) >= 0.15 for _, _, _, rate in skew):
            tags.append("sales_route_strength")
        p("")

    # ---- 対照比較: 主要指標で高スコア失注 vs 低スコア成約 vs 同帯 ----
    p("## クロス比較（コホート間 主要指標 mean）")
    p("")
    p("| 指標 | A:高スコア失注 | B:低スコア成約 | C:同帯成約 | C:同帯失注 |")
    p("|---|---|---|---|---|")
    def mv(s, k):
        d = s.get(k)
        return f"{d['mean']:.2f}" if d else "-"
    for k, lab in [("score", "スコア"), ("op_margin", "営業利益率%"),
                   ("rent_to_sales", "リース料/売上%"), ("debt_to_sales", "総借入/売上%"),
                   ("asset_score", "物件スコア"), ("nenshu", "売上高")]:
        p(f"| {lab} | {mv(sA,k)} | {mv(sB,k)} | {mv(sCw,k)} | {mv(sCl,k)} |")
    p("")

    p("## 検出された発見タグ（仮付与）")
    p("")
    if tags:
        for t in tags:
            p(f"- ✅ `{t}`")
    else:
        p("- （統計的に有意な閾値を超えるタグはなし）")
    p("")

    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(L))
    print("WROTE", OUT)
    print("TAGS", tags)
    print("A/B/Cw/Cl counts:", len(high_score_lost), len(low_score_won), len(same_won), len(same_lost))


if __name__ == "__main__":
    main()
