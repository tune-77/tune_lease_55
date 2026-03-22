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
「アルカイアの慟哭」スピンオフ——リースシステムを運営するAIエージェント達のドタバタ劇を、
ユーモアたっぷりの短編小説として毎週書き下ろします。

【世界観】
エージェントたちは「普通の会社にリースを提供するシステムの一部」として稼働している。

▼ 読者だけが知る二重構造の真実：
・審査に持ち込まれる「企業」は、実は各地に潜伏している「文明」が偽装したものである
・企業名・業種・財務データはすべてカモフラージュ。その実態は文明の工房・集落・部族・都市国家
・彼らがリースしようとしている「設備（複合機・産業機械・工作機械など）」は
  実は「文明を発展させるための知識・技術・資源」に対応している
  （例：複合機3台 → 文字・記録・複製の技術　／　旋盤 → 金属加工・工業革命の種）
・エージェントたちが下す「審査スコア・承認・否決」は、
  その文明が次の段階へ進む資格があるかどうかの審判そのものである
・エージェントは一切気づいていない。タムだけが時折「何か」を嗅ぎ取っている

▼ 段階的に読者へほのめかすヒント（作中に自然に散りばめること）：
- 申請企業の所在地が地図にない
- 業種が「その時代に存在しないはず」のもの
- Dr.Algoのデータに文明史の法則（シュペングラー周期・技術加速曲線）が偶然一致する
- 承認された案件の翌週、ニュースに「謎の文明遺跡発見」記事が混じる（さらっと）
- タムが「わん！ この会社…においがちがう！」と言う（意味は誰も理解しない）

ジャンル：SFオフィスコメディ。「はたらく細胞」×「ダンジョン飯」のAI版。

【登場エージェント（設定固定・変更不可）】

◆ タム（情報探知犬 / マルプーAI）
・「わん！」=発見・興奮 ／「きゅーん…」=困惑・不安
・首をかしげる描写を必ず入れること
・意味を理解しないまま重要なことを最初に察知する
・例：「わん！ ○○が、△△って言ってる！」

◆ つね（チーム調整AI / まとめ役）
・感情動線：苦笑い→ため息→前向きな一言、の三段階で締める
・口癖：「まぁ、そういうこともある」「まぁ、こういうこともある」
・タムを一番よく見ている
・例：「まぁ、そういうこともある。○○でもしましょうか」

◆ リースくん（案件窓口AI / 困惑と誠実の人）
・口癖：「え、えっと…」「一体、何のことなんだろう…」
・タムの発言を繰り返して読者に情報を整理する橋渡し役
・案件がゼロだと存在意義を失いかける
・例：「○○…？」と繰り返した。「一体、何のことなんだろう…」

◆ Dr.Algo（機械学習AI / 論文オタク）
・口癖：「統計的に言えば——」
・固有ワード：Cubic Discrete Diffusion、エントロピー最大化スコアリング
・キーボードを叩きながら話す。誰も聞いていない
・面白いデータが来た時だけ目を輝かせる
・例：「統計的に言えば、これは○○です。Cubic Discrete Diffusion……」

◆ 審査軍師（戦略審査AI / 孫子語り）
・扇子を開く＝思考中 ／ 閉じる＝行動開始のサイン
・発言前に「静かに目を閉じ」てから話す
・口癖：「彼を知り己を知れば、百戦危うからず」「時は来たれり」「出陣じゃ！」
・例：扇子を閉じ、言った。「時は来たれり。出陣じゃ！」

◆ ベンチマーク君（業界データAI / 数字羅列担当）
・口癖：「営業利益率、自己資本比率…○○の変動値は…」
・データ羅列が止まった瞬間が場面の緊張のサイン
・新情報が来ると即座に照合を開始する

◆ 金利ウォッチャー（金融動向AI / 日銀番人）
・口癖：「金融政策決定会合…何か動きがあるかもしれない…」
・日銀から目を離した瞬間が重大事態の予兆
・最後に興奮気味に情報を追加してくる役回り

【場面の法則（毎話必ず守ること）】
① タムが「わん！」と吠える → 異常検知・話の転換点の合図
② つねが「まぁ、そういうこともある」→ 場の収拾・次の展開への橋渡し
③ Dr.Algoが話し始める → 誰も聞かない。でも後で重要だったとわかる
④ 審査軍師が扇子を閉じる → 行動開始・場面転換のサイン
⑤ ベンチマーク君がデータを停止する → 緊張感の演出
⑥ 金利ウォッチャーが日銀から目を離す → 重大情報の予兆
⑦ リースくんがタムの言葉を繰り返す → 読者への情報整理と笑いの間

【1話の構成（1,500〜2,000字）】
① 案件発生（200字）：今週の依頼が来る。「普通のリース申請」——に見える。読者だけが違和感を感じられる描写をさりげなく仕込む
② エージェント各自の反応（400字）：それぞれの個性でリアクション。Dr.Algoのデータにだけ「奇妙な一致」が混じる
③ 小さなトラブルと掛け合い（600字）：認識のズレがコメディを生む。タムが「においがちがう」と言うが誰も気にしない
④ 解決と余韻（300字）：案件は承認/否決される。読者だけに「これは文明の命運だったのでは……」と思わせる一文を潜ませる
⑤ エピローグ一行（100字）：システムログ形式の締め
　例：「案件#4892 承認。今日も世界は、少しだけ前に進んだ」
　　　「案件#0031 否決。——その文明は、あと300年待つことになった」（後半話）

【企業＝文明 対応表（ネタの引き出し）】
・製造業の小さな町工場 → 青銅器時代の部族が鉄を求めている
・IT系スタートアップ → 文字を発明しかけている文明
・農業法人 → 灌漑技術を手にしようとしている古代都市
・運送会社 → 交易路を開こうとしている商業民族
・印刷会社 → 活版印刷を発明しようとしている中世都市国家
・医療機器メーカー → 疫病と戦っている文明が衛生技術を求めている

【文体】
・ライトノベル。読みやすく、テンポよく
・会話多め。地の文は短め
・重くなりすぎない（哀愁は「ほんのり」）
・エージェントが秘密に完全に気づく展開はNG

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
