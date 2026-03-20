# -*- coding: utf-8 -*-
"""
components/subsidy_master.py
=============================
補助金マスタ管理 — 業種 × 設備種別 で補助金を自動マッチングし
審査フォームで「使える補助金」をカード表示する。

## DB テーブル: subsidy_master
| カラム | 型 | 説明 |
|--------|-----|------|
| id | INTEGER PK | |
| name | TEXT | 補助金名称 |
| max_amount | INTEGER | 上限額（万円） |
| industry_codes | TEXT | 適用業種コード（カンマ区切り、空=全業種） |
| asset_keywords | TEXT | 適用設備キーワード（カンマ区切り、空=全設備） |
| deadline | TEXT | 申請期限（YYYY-MM-DD or "随時"） |
| url | TEXT | 公式URL |
| notes | TEXT | 備考 |
| active | INTEGER | 1=有効 0=無効 |
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import date
from typing import Optional

import streamlit as st

# ── パス設定 ─────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_DIR)
_DB_PATH = os.path.join(_BASE, "data", "lease_data.db")

# ── 初期補助金データ（手動更新のベースライン） ──────────────────────────────
_INITIAL_SUBSIDIES = [
    {
        "name": "ものづくり補助金",
        "max_amount": 1250,
        "industry_codes": "09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26",
        "asset_keywords": "機械,設備,ロボット,NC,CNC,旋盤,マシニング,プレス",
        "deadline": "随時",
        "url": "https://portal.monodukuri-hojo.jp/",
        "notes": "製造業・IT業等の設備投資。補助率1/2〜2/3",
        "active": 1,
    },
    {
        "name": "IT導入補助金",
        "max_amount": 450,
        "industry_codes": "",
        "asset_keywords": "PC,サーバー,IT,ソフトウェア,システム,クラウド",
        "deadline": "随時",
        "url": "https://www.it-hojo.jp/",
        "notes": "全業種対象。ITツール導入費用の補助",
        "active": 1,
    },
    {
        "name": "事業再構築補助金",
        "max_amount": 7000,
        "industry_codes": "",
        "asset_keywords": "",
        "deadline": "随時",
        "url": "https://jigyou-saikouchiku.jp/",
        "notes": "新分野展開・業態転換等。補助率1/2〜2/3",
        "active": 1,
    },
    {
        "name": "省エネルギー設備導入補助金",
        "max_amount": 3000,
        "industry_codes": "",
        "asset_keywords": "省エネ,エコ,コンプレッサー,ボイラー,空調,冷凍,LED",
        "deadline": "随時",
        "url": "https://sii.or.jp/",
        "notes": "省エネ効果が確認できる設備が対象",
        "active": 1,
    },
    {
        "name": "農業次世代人材投資資金",
        "max_amount": 500,
        "industry_codes": "01,02",
        "asset_keywords": "農機,トラクター,コンバイン,田植機,農業機械",
        "deadline": "随時",
        "url": "https://www.maff.go.jp/",
        "notes": "農業従事者向け。農業機械の導入費用補助",
        "active": 1,
    },
    {
        "name": "医療機器導入補助（地方版）",
        "max_amount": 500,
        "industry_codes": "83",
        "asset_keywords": "医療機器,CT,MRI,レントゲン,超音波,内視鏡",
        "deadline": "随時",
        "url": "",
        "notes": "都道府県・市区町村ごとに内容が異なる。要確認",
        "active": 1,
    },
    {
        "name": "GX（グリーントランスフォーメーション）補助金",
        "max_amount": 1500,
        "industry_codes": "",
        "asset_keywords": "EV,電気自動車,燃料電池,FCV,水素,太陽光,蓄電池",
        "deadline": "随時",
        "url": "https://gx-league.go.jp/",
        "notes": "脱炭素・GX投資が対象",
        "active": 1,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# DB 操作
# ══════════════════════════════════════════════════════════════════════════════

def init_subsidy_table() -> None:
    """subsidy_master テーブルを初期化（なければ作成＋初期データ投入）。"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subsidy_master (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                max_amount      INTEGER NOT NULL DEFAULT 0,
                industry_codes  TEXT    NOT NULL DEFAULT '',
                asset_keywords  TEXT    NOT NULL DEFAULT '',
                deadline        TEXT    NOT NULL DEFAULT '随時',
                url             TEXT    NOT NULL DEFAULT '',
                notes           TEXT    NOT NULL DEFAULT '',
                active          INTEGER NOT NULL DEFAULT 1
            )
        """)
        # 初期データが未挿入なら投入
        count = conn.execute("SELECT COUNT(*) FROM subsidy_master").fetchone()[0]
        if count == 0:
            conn.executemany("""
                INSERT INTO subsidy_master
                    (name, max_amount, industry_codes, asset_keywords, deadline, url, notes, active)
                VALUES (:name, :max_amount, :industry_codes, :asset_keywords,
                        :deadline, :url, :notes, :active)
            """, _INITIAL_SUBSIDIES)
        conn.commit()


def load_subsidies(active_only: bool = True) -> list[dict]:
    """補助金マスタを全件取得。"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        where = "WHERE active = 1" if active_only else ""
        rows = conn.execute(
            f"SELECT * FROM subsidy_master {where} ORDER BY max_amount DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def save_subsidy(data: dict) -> int:
    """補助金レコードを新規登録し ID を返す。"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        cur = conn.execute("""
            INSERT INTO subsidy_master
                (name, max_amount, industry_codes, asset_keywords, deadline, url, notes, active)
            VALUES (:name, :max_amount, :industry_codes, :asset_keywords,
                    :deadline, :url, :notes, :active)
        """, data)
        row_id = cur.lastrowid
        conn.commit()
    return row_id


def update_subsidy(subsidy_id: int, data: dict) -> None:
    """既存レコードを更新する。"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.execute("""
            UPDATE subsidy_master SET
                name=:name, max_amount=:max_amount, industry_codes=:industry_codes,
                asset_keywords=:asset_keywords, deadline=:deadline, url=:url,
                notes=:notes, active=:active
            WHERE id=:id
        """, {**data, "id": subsidy_id})
        conn.commit()


def delete_subsidy(subsidy_id: int) -> None:
    """論理削除（active=0）。"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.execute("UPDATE subsidy_master SET active=0 WHERE id=?", (subsidy_id,))
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# マッチングロジック
# ══════════════════════════════════════════════════════════════════════════════

def match_subsidies(
    industry_code: Optional[str],
    asset_name: Optional[str],
    max_results: int = 3,
) -> list[dict]:
    """
    業種コードと設備名称から使える補助金を最大 max_results 件返す。

    Args:
        industry_code: 業種コード先頭2桁（例 "44"）。None なら全業種対象のみ。
        asset_name: 設備名称（例 "NC旋盤"）。None なら設備キーワード無視。
        max_results: 返す件数の上限。

    Returns:
        マッチした補助金の dict リスト（max_amount 降順）。
    """
    all_subsidies = load_subsidies(active_only=True)
    matched: list[dict] = []

    for sub in all_subsidies:
        # 業種チェック
        codes = [c.strip() for c in sub["industry_codes"].split(",") if c.strip()]
        if codes and industry_code:
            industry_match = any(industry_code.startswith(c) or c.startswith(industry_code[:2]) for c in codes)
        elif codes:
            industry_match = False  # コード指定あり・入力なし → スキップ
        else:
            industry_match = True  # 全業種対象

        if not industry_match:
            continue

        # 設備キーワードチェック
        keywords = [k.strip() for k in sub["asset_keywords"].split(",") if k.strip()]
        if keywords and asset_name:
            asset_match = any(k in asset_name for k in keywords)
        elif keywords:
            asset_match = False  # キーワードあり・設備名なし → スキップ
        else:
            asset_match = True  # 全設備対象

        if not asset_match:
            continue

        matched.append(sub)
        if len(matched) >= max_results:
            break

    return matched


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

def render_subsidy_cards(industry_code: Optional[str], asset_name: Optional[str]) -> None:
    """
    審査フォームから呼び出す補助金カード表示。
    マッチした補助金を最大3件カード表示する。
    """
    init_subsidy_table()
    matches = match_subsidies(industry_code, asset_name, max_results=3)

    if not matches:
        return

    st.markdown("---")
    st.subheader("💰 使える可能性がある補助金")
    st.caption("業種・設備をもとに自動マッチングしました。申請要件は必ず公式サイトでご確認ください。")

    cols = st.columns(len(matches))
    for col, sub in zip(cols, matches):
        with col:
            amount_str = f"最大 **{sub['max_amount']:,}万円**" if sub["max_amount"] > 0 else "金額要確認"
            deadline_str = sub["deadline"] if sub["deadline"] else "随時"
            url = sub["url"]
            link = f"[公式サイト]({url})" if url else "（URL未登録）"
            st.markdown(
                f"""
<div style="border:1px solid #e0e0e0; border-radius:8px; padding:12px; background:#fafafa;">
<div style="font-size:0.85rem; font-weight:bold; margin-bottom:4px;">{sub['name']}</div>
<div style="font-size:1.1rem; color:#1a7f37;">{amount_str}</div>
<div style="font-size:0.75rem; color:#666; margin-top:4px;">期限: {deadline_str}</div>
<div style="font-size:0.75rem; color:#555; margin-top:4px;">{sub['notes'][:60]}{'…' if len(sub['notes']) > 60 else ''}</div>
<div style="font-size:0.75rem; margin-top:6px;">{link}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_subsidy_master_admin() -> None:
    """補助金マスタの管理画面（一覧・追加・削除）。"""
    init_subsidy_table()
    st.subheader("💰 補助金マスタ管理")
    st.caption("業種コード・設備キーワードを登録すると、審査フォームで自動マッチングされます。")

    # ── 一覧 ──────────────────────────────────────────────────────────────────
    subsidies = load_subsidies(active_only=False)
    if subsidies:
        import pandas as pd
        df = pd.DataFrame(subsidies)[
            ["id", "name", "max_amount", "industry_codes", "asset_keywords", "deadline", "active"]
        ].rename(columns={
            "id": "ID", "name": "補助金名", "max_amount": "上限(万円)",
            "industry_codes": "業種コード", "asset_keywords": "設備KW",
            "deadline": "期限", "active": "有効",
        })
        df["有効"] = df["有効"].map({1: "✅", 0: "❌"})
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 削除
        del_id = st.number_input("無効化する ID", min_value=1, step=1, key="subsidy_del_id")
        if st.button("🗑️ 無効化", key="subsidy_del_btn"):
            delete_subsidy(int(del_id))
            st.success(f"ID {del_id} を無効化しました。")
            st.rerun()
    else:
        st.info("登録されていません。")

    # ── 新規追加フォーム ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**新規追加**")
    with st.form("subsidy_add_form"):
        name        = st.text_input("補助金名称", placeholder="ものづくり補助金")
        max_amount  = st.number_input("上限額（万円）", min_value=0, step=50)
        industry_codes = st.text_input("業種コード（カンマ区切り、空=全業種）", placeholder="09,10,11")
        asset_keywords = st.text_input("設備キーワード（カンマ区切り、空=全設備）", placeholder="機械,ロボット")
        deadline    = st.text_input("申請期限", value="随時")
        url         = st.text_input("公式URL")
        notes       = st.text_area("備考", height=60)
        submitted   = st.form_submit_button("追加")

    if submitted and name:
        save_subsidy({
            "name": name, "max_amount": int(max_amount),
            "industry_codes": industry_codes, "asset_keywords": asset_keywords,
            "deadline": deadline, "url": url, "notes": notes, "active": 1,
        })
        st.success(f"「{name}」を登録しました。")
        st.rerun()
