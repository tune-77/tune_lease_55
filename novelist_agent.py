# -*- coding: utf-8 -*-
"""
novelist_agent.py
=================
文豪AI「波乱丸（はらんまる）」— リースシステムのエージェント達が繰り広げる
ドタバタ劇を、ユーモアたっぷりの短編小説として毎週火曜日に書き下ろす。

登場人物（エージェント）:
  - つね     : 統括マネージャー。冷静沈着を装うが実は情に厚い。
  - タム     : 謎の子犬AI。無邪気に見えて全てを見通す。マルプー。
  - Dr.Algo  : 数学者AI。AUCとカルマンフィルタしか頭にない理系オタク。
  - 軍師      : 審査軍師。孫子を引用しがち。
  - リースくん: 審査ウィザード担当の真面目な新人AI。
  - その他    : ベンチマーク君、金利ウォッチャーなど脇役多数。
"""
from __future__ import annotations

import os
import sqlite3
import json
import datetime

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_NOVEL_DB    = os.path.join(_BASE_DIR, "data", "novelist_agent.db")
_LEASE_DB    = os.path.join(_BASE_DIR, "data", "lease_data.db")
_HUB_LOG     = os.path.join(_BASE_DIR, "data", "agent_hub_log.jsonl")

# ══════════════════════════════════════════════════════════════════════════════
# DB 初期化
# ══════════════════════════════════════════════════════════════════════════════

def init_novel_db() -> None:
    conn = sqlite3.connect(_NOVEL_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS novels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            week_label  TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            body        TEXT    NOT NULL,
            episode_no  INTEGER DEFAULT 1
        );
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# データ収集
# ══════════════════════════════════════════════════════════════════════════════

def _collect_recent_screenings(n: int = 5) -> list[dict]:
    """最近の審査案件を取得してネタにする。"""
    try:
        conn = sqlite3.connect(_LEASE_DB)
        rows = conn.execute("""
            SELECT company_name, industry, total_score, grade, created_at
            FROM screening_results
            ORDER BY created_at DESC
            LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [
            {"company": r[0] or "某社", "industry": r[1] or "不明業種",
             "score": r[2], "grade": r[3], "date": r[4]}
            for r in rows
        ]
    except Exception:
        return []


def _collect_hub_events(n: int = 10) -> list[dict]:
    """エージェントハブのログから最近のイベントを取得してネタにする。"""
    events = []
    try:
        if not os.path.exists(_HUB_LOG):
            return events
        with open(_HUB_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines[-n:]):
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass
    except Exception:
        pass
    return events


def _collect_math_discoveries(n: int = 3) -> list[str]:
    """数学者エージェントの最新発見をネタにする。"""
    math_db = os.path.join(_BASE_DIR, "data", "math_discoveries.db")
    try:
        conn = sqlite3.connect(math_db)
        rows = conn.execute(
            "SELECT method_name, field_tag FROM math_discoveries ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
        return [f"{r[0]}（{r[1]}）" for r in rows]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 小説生成
# ══════════════════════════════════════════════════════════════════════════════

_NOVELIST_SYSTEM = """あなたは文豪AI「波乱丸」です。
リース審査AIシステムの中で働くエージェント達のドタバタ劇を、
ユーモアたっぷりの短編小説として書きます。

【登場キャラクター】
- つね     : 統括マネージャー。冷静沈着を装うが情に厚い女傑。口癖「まぁ、そういうこともある」
- タム     : 謎の子犬AIマルプー。「わん！」と言いながら本質を突く。秘密が多い
- Dr.Algo  : 数学者AI。AUCとカルマンフィルタしか頭にない。口癖「統計的に言えば…」
- 審査軍師  : 孫子を引用しがちな老練なAI。「彼を知り己を知れば…」が口癖
- リースくん: 真面目な新人審査AI。困ると「え、えっと…」と言いながら頑張る
- ベンチマーク君: 業界データを集め続ける几帳面なAI。数字に異常に詳しい
- 金利ウォッチャー: 日銀の動向に一喜一憂するAI。常に緊張気味

【執筆スタイル】
- 800〜1200文字の短編小説
- 会話文を多用してテンポよく
- ユーモアと人情を織り交ぜる
- リース審査の専門用語をさりげなく盛り込む
- 毎回タイトルを工夫する（例：「AUCの涙」「タムの謎の予言」など）
- 末尾に「つづく」か「完」をつける

出力形式：
タイトル：【タイトル】
本文：（小説本文）
"""


def generate_novel(episode_no: int = None, custom_theme: str = "") -> dict:
    """
    今週の短編小説を生成して DB に保存。
    Returns {"title": str, "body": str, "week_label": str, "episode_no": int}
    """
    init_novel_db()

    # エピソード番号
    if episode_no is None:
        conn = sqlite3.connect(_NOVEL_DB)
        last = conn.execute("SELECT MAX(episode_no) FROM novels").fetchone()[0]
        conn.close()
        episode_no = (last or 0) + 1

    now = datetime.datetime.now()
    week_label = now.strftime("第%Y年%m月%d日号")

    # ネタ収集
    screenings = _collect_recent_screenings(5)
    hub_events = _collect_hub_events(10)
    math_hits  = _collect_math_discoveries(3)

    # ネタまとめ
    neta_lines = [f"第{episode_no}話の執筆をお願いします。今週のネタ："]

    if screenings:
        neta_lines.append("\n【今週の審査案件（ネタ素材）】")
        for s in screenings:
            neta_lines.append(
                f"  ・{s['company']}（{s['industry']}）スコア{s['score']}点 {s['grade']}"
            )
    else:
        neta_lines.append("\n【今週の審査】案件データなし（エージェント達の日常でOK）")

    if hub_events:
        neta_lines.append("\n【最近のエージェント活動ログ（ネタ素材）】")
        for ev in hub_events[:5]:
            agent = ev.get("agent", "?")
            action = ev.get("action", "?")
            detail = ev.get("detail", "")
            neta_lines.append(f"  ・{agent}が「{action}」→ {detail}")

    if math_hits:
        neta_lines.append("\n【Dr.Algoが最近研究していた手法】")
        for m in math_hits:
            neta_lines.append(f"  ・{m}")

    if custom_theme:
        neta_lines.append(f"\n【今回の特別テーマ】{custom_theme}")

    prompt = "\n".join(neta_lines)

    # AI呼び出し
    try:
        from ai_chat import _chat_for_thread, is_ai_available
        import streamlit as st
        from components.agent_hub import _get_ai_settings

        if not is_ai_available():
            return _fallback_novel(episode_no, week_label)

        engine, model, api_key, gemini_model = _get_ai_settings()
        messages = [
            {"role": "system", "content": _NOVELIST_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        raw = _chat_for_thread(engine, model, messages,
                               timeout_seconds=120,
                               api_key=api_key,
                               gemini_model=gemini_model)
        text = (raw.get("message") or {}).get("content", "") or ""
    except Exception as e:
        text = f"[小説生成エラー: {e}]"

    # タイトルと本文を分離
    title = f"第{episode_no}話"
    body  = text
    if "タイトル：" in text:
        parts = text.split("タイトル：", 1)
        rest  = parts[1]
        if "\n" in rest:
            title_line, body = rest.split("\n", 1)
            title = title_line.strip().lstrip("【").rstrip("】")
            body  = body.replace("本文：", "").strip()
        else:
            title = rest.strip().lstrip("【").rstrip("】")
    elif "【" in text and "】" in text:
        import re
        m = re.search(r"【(.+?)】", text)
        if m:
            title = m.group(1)

    # DB保存
    conn = sqlite3.connect(_NOVEL_DB)
    conn.execute(
        "INSERT INTO novels (ts, week_label, title, body, episode_no) VALUES (?,?,?,?,?)",
        (now.isoformat(), week_label, title, body, episode_no)
    )
    conn.commit()
    conn.close()

    return {"title": title, "body": body, "week_label": week_label, "episode_no": episode_no}


def _fallback_novel(episode_no: int, week_label: str) -> dict:
    """AI未設定時のサンプル小説。"""
    title = "タムの怪しい月曜日"
    body  = """　月曜の朝、オフィスに最初にやってきたのはタムだった。
「わん！」
　タムはそう一声吠えると、スコアリングDBを覗き込んだ。今週も審査案件が積み上がっている。

「統計的に言えば」とDr.Algoが割り込んだ。「月曜朝のAUCは0.87まで下がる傾向がある」
「それ、ぼくが眠いだけじゃないですか？」リースくんが恐る恐る反論した。

「まぁ、そういうこともある」
　つねが珈琲を片手に現れた。その一言で場が静まり返る。

　審査軍師がおもむろに口を開いた。「孫子曰く、戦わずして勝つのが上策。今日の案件、スコア67点——要審議ゾーンじゃな」
「わんっ！（訳：つまり、今日も残業ってこと？）」

　タムの尻尾がくるりと巻いた。今週も長い一週間が始まろうとしていた。

完"""
    init_novel_db()
    now = datetime.datetime.now()
    conn = sqlite3.connect(_NOVEL_DB)
    conn.execute(
        "INSERT INTO novels (ts, week_label, title, body, episode_no) VALUES (?,?,?,?,?)",
        (now.isoformat(), week_label, title, body, episode_no)
    )
    conn.commit()
    conn.close()
    return {"title": title, "body": body, "week_label": week_label, "episode_no": episode_no}


def load_novels(limit: int = 20) -> list[dict]:
    """過去の小説一覧を新しい順で返す。"""
    init_novel_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute(
        "SELECT id, ts, week_label, title, body, episode_no FROM novels ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "ts": r[1], "week_label": r[2],
         "title": r[3], "body": r[4], "episode_no": r[5]}
        for r in rows
    ]


def get_latest_novel() -> dict | None:
    """最新話を返す。なければ None。"""
    novels = load_novels(1)
    return novels[0] if novels else None
