"""
SQLite past_cases テーブルから自然言語クエリに対応した統計を構築するモジュール。
めぶきちゃんチャットのDB直接分析機能を担当する。
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from typing import Optional

_DB_PATH: Optional[str] = None


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        import os
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.dirname(_here)
        _DB_PATH = os.path.join(_root, "data", "lease_data.db")
    return _DB_PATH


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


# ── 意図判定 ──────────────────────────────────────────────────────────────────

_DB_INTENT_KEYWORDS = [
    "データ", "分析して", "推論して", "格付", "失注", "成約",
    "件数", "統計", "傾向", "どの業種", "何件", "スコア",
    "実績", "過去案件", "パターン", "割合", "比率", "ランキング",
    "data", "db", "業界別", "業種別", "案件を", "案件の",
    "成約率", "失注率",
]

_GRADE_KEYWORDS = ["格付", "grade", "ランク", "格"]
_LOST_KEYWORDS = ["失注", "負け", "断られ", "取れなかった", "失敗"]
_WON_KEYWORDS = ["成約", "勝ち", "取れた", "承認", "通った"]
_INDUSTRY_KEYWORDS = ["業界", "業種", "産業", "どの業"]
_SCORE_KEYWORDS = ["スコア", "点数", "score", "得点"]


def _has_db_intent(message: str) -> bool:
    return any(k in message for k in _DB_INTENT_KEYWORDS)


# ── メインエントリ ─────────────────────────────────────────────────────────────

def build_db_context(message: str) -> str:
    """
    ユーザーメッセージに応じたDB統計テキストを返す。
    チャットのシステムプロンプトに追加して使う。
    空文字を返した場合は注入しない。
    """
    if not _has_db_intent(message):
        return ""

    try:
        sections: list[str] = []

        # 全体サマリーは常に含める
        summary = _build_summary_stats()
        if summary:
            sections.append(summary)

        # 格付フィルタを解析
        grade_filter = _parse_grade_filter(message)

        if any(k in message for k in _GRADE_KEYWORDS) or grade_filter:
            grade_sec = _build_grade_stats(message)
            if grade_sec:
                sections.append(grade_sec)

        # 失注分析
        if any(k in message for k in _LOST_KEYWORDS):
            lost_sec = _build_lost_analysis(grade_filter=grade_filter)
            if lost_sec:
                sections.append(lost_sec)
        # 成約分析
        elif any(k in message for k in _WON_KEYWORDS):
            won_sec = _build_won_analysis(grade_filter=grade_filter)
            if won_sec:
                sections.append(won_sec)

        # 業種別統計（格付フィルタ付きも可）
        industry_sec = _build_industry_stats(grade_filter=grade_filter)
        if industry_sec:
            sections.append(industry_sec)

        if not sections:
            return ""

        header = "【過去案件DB統計 — 実データに基づく分析】"
        body = "\n\n".join(sections)
        footer = (
            "※ 上記は lease_data.db の past_cases テーブルから集計した実績値です。"
            "Obsidianノートではなく実際の案件データです。"
        )
        return f"\n\n{header}\n{body}\n\n{footer}"

    except Exception as exc:
        return f"\n\n【DB統計取得中にエラーが発生しました: {exc}】"


# ── 格付フィルタ解析 ──────────────────────────────────────────────────────────

_GRADE_MAP = {
    "1-3": ["1-3", "1〜3", "1~3", "一〜三"],
    "4-6": ["4-6", "4〜6", "4~6", "四〜六", "2〜5", "2-5", "2~5"],
    "要注意": ["要注意", "watch"],
    "無格付": ["無格付", "none", "無評価"],
}


def _parse_grade_filter(message: str) -> Optional[str]:
    """メッセージから格付フィルタ（'1-3','4-6','要注意','無格付'）を推測する。"""
    for grade, triggers in _GRADE_MAP.items():
        if any(t in message for t in triggers):
            return grade
    return None


# ── 集計関数群 ────────────────────────────────────────────────────────────────

def _build_summary_stats() -> str:
    with closing(_open_db()) as conn:
        row = conn.execute("""
            SELECT
                count(*) AS total,
                sum(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) AS won,
                sum(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS lost,
                round(avg(score), 1) AS avg_score,
                round(min(score), 1) AS min_score,
                round(max(score), 1) AS max_score
            FROM past_cases
        """).fetchone()
    if not row or not row["total"]:
        return ""
    total = row["total"] or 0
    won = row["won"] or 0
    lost = row["lost"] or 0
    won_rate = round(won / total * 100, 1) if total else 0
    return (
        "■ 全体サマリー\n"
        f"  総案件数: {total}件\n"
        f"  成約: {won}件 ({won_rate}%) / 失注: {lost}件\n"
        f"  スコア分布: 平均{row['avg_score']} / 最低{row['min_score']} / 最高{row['max_score']}"
    )


def _build_industry_stats(grade_filter: Optional[str] = None) -> str:
    """業種別成約率・件数・平均スコアを返す（格付フィルタ対応）。"""
    with closing(_open_db()) as conn:
        if grade_filter:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS cnt,
                    round(avg(score), 1) AS avg_score,
                    sum(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) AS won,
                    sum(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS lost
                FROM past_cases
                WHERE json_extract(data, '$.inputs.grade') = ?
                  AND industry_sub != '' AND industry_sub != '0'
                GROUP BY industry_sub
                ORDER BY cnt DESC
                LIMIT 12
            """, (grade_filter,)).fetchall()
            label = f"■ 業種別統計（格付:{grade_filter}、件数上位12業種）"
        else:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS cnt,
                    round(avg(score), 1) AS avg_score,
                    sum(CASE WHEN final_status IN ('成約','検収完了') THEN 1 ELSE 0 END) AS won,
                    sum(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS lost
                FROM past_cases
                WHERE industry_sub != '' AND industry_sub != '0'
                GROUP BY industry_sub
                ORDER BY cnt DESC
                LIMIT 12
            """).fetchall()
            label = "■ 業種別統計（件数上位12業種）"

    if not rows:
        return ""
    lines = [label]
    for r in rows:
        total = r["cnt"] or 0
        won_rate = round((r["won"] or 0) / total * 100, 1) if total else 0
        lost_rate = round((r["lost"] or 0) / total * 100, 1) if total else 0
        lines.append(
            f"  {r['industry_sub']}: {total}件 | 成約率{won_rate}% | 失注率{lost_rate}% | 平均スコア{r['avg_score']}"
        )
    return "\n".join(lines)


def _build_grade_stats(message: str) -> str:
    """格付別成約・失注件数・平均スコアを返す。"""
    with closing(_open_db()) as conn:
        rows = conn.execute(
            "SELECT data, final_status, score FROM past_cases LIMIT 5000"
        ).fetchall()

    grade_buckets: dict[str, dict] = {}
    for r in rows:
        try:
            d = json.loads(r["data"] or "{}")
            grade = (d.get("inputs") or {}).get("grade") or "不明"
        except Exception:
            grade = "不明"
        if grade not in grade_buckets:
            grade_buckets[grade] = {"total": 0, "won": 0, "lost": 0, "scores": []}
        b = grade_buckets[grade]
        b["total"] += 1
        status = r["final_status"] or ""
        if status in ("成約", "検収完了"):
            b["won"] += 1
        elif status == "失注":
            b["lost"] += 1
        if r["score"] is not None:
            b["scores"].append(float(r["score"]))

    if not grade_buckets:
        return ""

    lines = ["■ 格付別統計"]
    grade_order = ["1-3", "4-6", "要注意", "無格付", "不明"]
    ordered = sorted(
        grade_buckets.items(),
        key=lambda kv: grade_order.index(kv[0]) if kv[0] in grade_order else 99,
    )
    for grade, b in ordered:
        total = b["total"]
        if total == 0:
            continue
        won_rate = round(b["won"] / total * 100, 1)
        lost_rate = round(b["lost"] / total * 100, 1)
        avg_score = round(sum(b["scores"]) / len(b["scores"]), 1) if b["scores"] else 0
        lines.append(
            f"  格付 {grade}: {total}件 | 成約率{won_rate}% | 失注率{lost_rate}% | 平均スコア{avg_score}"
        )
    return "\n".join(lines)


def _build_lost_analysis(grade_filter: Optional[str] = None) -> str:
    """失注案件の業種別・スコア分布分析を返す。"""
    with closing(_open_db()) as conn:
        if grade_filter:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS lost_cnt,
                    round(avg(score), 1) AS avg_score,
                    round(min(score), 1) AS min_score,
                    round(max(score), 1) AS max_score
                FROM past_cases
                WHERE final_status = '失注'
                  AND json_extract(data, '$.inputs.grade') = ?
                GROUP BY industry_sub
                ORDER BY lost_cnt DESC
                LIMIT 10
            """, (grade_filter,)).fetchall()
            label = f"■ 失注分析（格付:{grade_filter}、業種別）"
        else:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS lost_cnt,
                    round(avg(score), 1) AS avg_score,
                    round(min(score), 1) AS min_score,
                    round(max(score), 1) AS max_score
                FROM past_cases
                WHERE final_status = '失注'
                GROUP BY industry_sub
                ORDER BY lost_cnt DESC
                LIMIT 10
            """).fetchall()
            label = "■ 失注分析（業種別）"

    if not rows:
        return ""
    lines = [label]
    for r in rows:
        lines.append(
            f"  {r['industry_sub']}: 失注{r['lost_cnt']}件 | 平均スコア{r['avg_score']}"
            f" (min:{r['min_score']} ~ max:{r['max_score']})"
        )
    return "\n".join(lines)


def _build_won_analysis(grade_filter: Optional[str] = None) -> str:
    """成約案件の業種別・スコア分布分析を返す。"""
    with closing(_open_db()) as conn:
        if grade_filter:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS won_cnt,
                    round(avg(score), 1) AS avg_score
                FROM past_cases
                WHERE final_status IN ('成約','検収完了')
                  AND json_extract(data, '$.inputs.grade') = ?
                GROUP BY industry_sub
                ORDER BY won_cnt DESC
                LIMIT 10
            """, (grade_filter,)).fetchall()
            label = f"■ 成約分析（格付:{grade_filter}、業種別）"
        else:
            rows = conn.execute("""
                SELECT
                    industry_sub,
                    count(*) AS won_cnt,
                    round(avg(score), 1) AS avg_score
                FROM past_cases
                WHERE final_status IN ('成約','検収完了')
                GROUP BY industry_sub
                ORDER BY won_cnt DESC
                LIMIT 10
            """).fetchall()
            label = "■ 成約分析（業種別）"

    if not rows:
        return ""
    lines = [label]
    for r in rows:
        lines.append(f"  {r['industry_sub']}: 成約{r['won_cnt']}件 | 平均スコア{r['avg_score']}")
    return "\n".join(lines)
