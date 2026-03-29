# -*- coding: utf-8 -*-
"""
novel_company_names.py
======================
小説AIで使用する架空企業名リスト（50社）。
SF・ビジネスドラマ・どちらのジャンルにも対応できるよう、
業種ごとにバランスよく収録。

使い方:
    from novel_company_names import NOVEL_COMPANIES, get_random_company
"""

import random

NOVEL_COMPANIES: list[dict] = [
    # ── 製造業 ─────────────────────────────────────────────────
    {"name": "株式会社北斗鉄工",         "industry": "製造業",   "tone": "business"},
    {"name": "有限会社丸善プレス",         "industry": "製造業",   "tone": "business"},
    {"name": "東亜精密工業株式会社",       "industry": "製造業",   "tone": "business"},
    {"name": "株式会社ゼノス・マニュファクチャ", "industry": "製造業", "tone": "sf"},
    {"name": "合同会社銀河鋳造所",         "industry": "製造業",   "tone": "sf"},
    {"name": "株式会社ミライ・フォージ",   "industry": "製造業",   "tone": "sf"},
    {"name": "有限会社山城板金",           "industry": "製造業",   "tone": "business"},
    {"name": "株式会社アーク・エンジニアリング", "industry": "製造業", "tone": "sf"},

    # ── 運送・物流 ──────────────────────────────────────────────
    {"name": "有限会社篠原運送",           "industry": "運送業",   "tone": "business"},
    {"name": "株式会社東海トラック",       "industry": "運送業",   "tone": "business"},
    {"name": "合同会社スピード急便",       "industry": "運送業",   "tone": "business"},
    {"name": "株式会社オービット・ロジスティクス", "industry": "運送業", "tone": "sf"},
    {"name": "有限会社黒潮配送",           "industry": "運送業",   "tone": "business"},
    {"name": "株式会社ネクサス・フレイト", "industry": "運送業",   "tone": "sf"},

    # ── 建設・土木 ──────────────────────────────────────────────
    {"name": "株式会社城南建設",           "industry": "建設業",   "tone": "business"},
    {"name": "有限会社大和土建",           "industry": "建設業",   "tone": "business"},
    {"name": "株式会社磐城工務店",         "industry": "建設業",   "tone": "business"},
    {"name": "合同会社テラ・ビルダーズ",   "industry": "建設業",   "tone": "sf"},
    {"name": "株式会社アトラス建工",       "industry": "建設業",   "tone": "sf"},
    {"name": "有限会社三河基礎工業",       "industry": "建設業",   "tone": "business"},

    # ── 医療・介護 ──────────────────────────────────────────────
    {"name": "医療法人社団 晴和会",       "industry": "医療・福祉", "tone": "business"},
    {"name": "社会福祉法人 光明苑",       "industry": "医療・福祉", "tone": "business"},
    {"name": "株式会社ヴィータ・メディカル", "industry": "医療・福祉", "tone": "sf"},
    {"name": "有限会社さくら介護センター", "industry": "医療・福祉", "tone": "business"},
    {"name": "株式会社ネオ・ライフケア",   "industry": "医療・福祉", "tone": "sf"},

    # ── IT・情報通信 ────────────────────────────────────────────
    {"name": "株式会社コアパルス・テック", "industry": "情報通信業", "tone": "sf"},
    {"name": "合同会社シナプス・データ",   "industry": "情報通信業", "tone": "sf"},
    {"name": "株式会社クォンタム・ソフト", "industry": "情報通信業", "tone": "sf"},
    {"name": "有限会社三条システム",       "industry": "情報通信業", "tone": "business"},
    {"name": "株式会社エクリプスIT",       "industry": "情報通信業", "tone": "sf"},

    # ── 飲食・サービス ──────────────────────────────────────────
    {"name": "株式会社大黒食品",           "industry": "飲食業",   "tone": "business"},
    {"name": "有限会社みのり農産",         "industry": "農業",     "tone": "business"},
    {"name": "合同会社サクラ・ダイニング", "industry": "飲食業",   "tone": "business"},
    {"name": "株式会社ノヴァ・フード・ラボ", "industry": "飲食業",  "tone": "sf"},
    {"name": "有限会社海の幸水産",         "industry": "漁業",     "tone": "business"},

    # ── 小売・卸売 ──────────────────────────────────────────────
    {"name": "株式会社マルヤマ商会",       "industry": "卸売業",   "tone": "business"},
    {"name": "有限会社東洋トレーディング", "industry": "卸売業",   "tone": "business"},
    {"name": "株式会社ヘリオス・マーケット", "industry": "小売業",  "tone": "sf"},
    {"name": "合同会社福島青果",           "industry": "小売業",   "tone": "business"},

    # ── エネルギー・環境 ────────────────────────────────────────
    {"name": "株式会社ソーラー・フロンティア東北", "industry": "電気・ガス", "tone": "sf"},
    {"name": "合同会社グリーン・パワー九州", "industry": "電気・ガス", "tone": "sf"},
    {"name": "株式会社ゲア・エナジー",     "industry": "電気・ガス", "tone": "sf"},
    {"name": "有限会社中部エコ産業",       "industry": "電気・ガス", "tone": "business"},

    # ── 不動産・リース ──────────────────────────────────────────
    {"name": "株式会社ブライト・プロパティ", "industry": "不動産業", "tone": "sf"},
    {"name": "有限会社和光不動産",         "industry": "不動産業", "tone": "business"},
    {"name": "合同会社南陽住建",           "industry": "不動産業", "tone": "business"},

    # ── SFのみ（文明・宇宙系） ──────────────────────────────────
    {"name": "コロニー自治体 アルテミス第三区", "industry": "公共", "tone": "sf"},
    {"name": "惑星開発公社 テラフォーム・ワン", "industry": "製造業", "tone": "sf"},
    {"name": "超弦工学研究所 株式会社",   "industry": "情報通信業", "tone": "sf"},
]


def get_random_company(industry: str | None = None, tone: str | None = None) -> dict:
    """
    ランダムに1社返す。
    industry / tone でフィルタリング可能。
    """
    pool = NOVEL_COMPANIES
    if industry:
        filtered = [c for c in pool if industry in c["industry"]]
        if filtered:
            pool = filtered
    if tone:
        filtered = [c for c in pool if c["tone"] == tone]
        if filtered:
            pool = filtered
    return random.choice(pool)


def get_company_names_text() -> str:
    """プロンプトに埋め込む用の企業名リスト文字列を返す。"""
    lines = [f"・{c['name']}（{c['industry']}）" for c in NOVEL_COMPANIES]
    return "\n".join(lines)
