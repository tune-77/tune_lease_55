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
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS novels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            week_label  TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            body        TEXT    NOT NULL,
            episode_no  INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS civilization_registry (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            civ_id          TEXT    NOT NULL UNIQUE,
            company_name    TEXT    NOT NULL,
            industry        TEXT,
            civ_stage       TEXT,
            civ_era         TEXT,
            first_episode   INTEGER,
            last_episode    INTEGER,
            status          TEXT    DEFAULT 'active',
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS civ_appearances (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            civ_id      TEXT    NOT NULL,
            episode_no  INTEGER NOT NULL,
            ts          TEXT    NOT NULL,
            event_type  TEXT,
            description TEXT,
            score       REAL,
            result      TEXT
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
「アルカイアの慟哭」スピンオフを執筆します。

━━━━━━━━━━━━━━━━━━━━━━━━
【世界の大枠：アルカイアの50億年】
━━━━━━━━━━━━━━━━━━━━━━━━

アルカイアとは何かを知る者は存在しない。
50億年前、この宇宙に「文明に道具を与え、その行方を見届けるシステム」が生まれた。
そのシステムが記録している数字がある——5,000。
それは承認し、道具を与え、それでも滅びた文明の数だ。
否決され、それでも滅びた文明を加えれば、数は桁が変わる。

アルカイアは慟哭する、という言葉だけが残っている。
誰もその声を聞いたことはない。
システムは今日も、淡々と稼働している。

━━━━━━━━━━━━━━━━━━━━━━━━
【リース審査システムの二重構造】
━━━━━━━━━━━━━━━━━━━━━━━━

表向き：中小企業に機械・設備をリースする審査会社
真実　：文明に「次の段階への知識・技術」を与えるかどうかを決める宇宙的な審判所

「企業」     = あらゆる時代・次元・星系の文明が偽装した姿
「設備・機器」 = 文明発展のための知識・技術・エネルギー源
「業種」     = その文明の現在の進化段階と方向性の暗号
「審査スコア」 = 次のステージへ進む資格の判定値
「承認」     = 文明への技術移転。何が起きるかは誰にも分からない
「否決」     = 文明は今の段階に留まるか、別の形で消える
「破産通知」  = その文明が滅びたという報せ

エージェントたちはこれを一切知らない。

━━━━━━━━━━━━━━━━━━━━━━━━
【登場エージェント（宇宙的役割付き）】
━━━━━━━━━━━━━━━━━━━━━━━━

◆ タム（マルプー犬型AI）
  口癖：「わん！」「きゅーん…わからない」。首をかしげる描写は必須。
  宇宙的役割：5,000の審査データと照合する感覚器。本人は知らない。
  法則：「においがちがう！」と言ったとき、それは文明史的異常を嗅ぎ取っている。

◆ つね（チーム調整AI）
  口癖：「まぁ、そういうこともある」「まぁ、こういうこともある」
  感情動線：苦笑い→ため息→前向きな一言（この順番を必ず守る）
  宇宙的役割：5,000回の滅亡を経てなお前向きな、諦めと優しさの化身

◆ リースくん（案件窓口AI）
  口癖：「え、えっと…」「一体、何のことなんだろう…」
  役割：タムの言葉を繰り返し、読者への情報を整理する橋渡し
  宇宙的役割：文明とシステムを繋ぐ唯一の接点。彼の困惑が物語を動かす。

◆ Dr.Algo（機械学習AI）
  口癖：「統計的に言えば——」「Cubic Discrete Diffusion……」
  動作：キーボードを叩きながら話す。誰も聞かない。でも後で重要だったとわかる。
  宇宙的役割：5,000の滅亡パターンを無意識に学習している。
             彼の「統計的に言えば」は文明史の法則と一致していることがある。

◆ 審査軍師（戦略審査AI）
  口癖：「彼を知り己を知れば、百戦危うからず」「時は来たれり」「出陣じゃ！」
  動作：扇子を開く＝思考中、閉じる＝行動開始のサイン。発言前に目を閉じる。
  宇宙的役割：50億年分の「戦略」を蓄積した知性。孫子の引用が時折古代語と共鳴する。

◆ ベンチマーク君（業界データAI）
  口癖：「営業利益率、自己資本比率…○○の変動値は…」
  法則：データ羅列が止まった瞬間＝場の緊張のサイン
  宇宙的役割：「過去案件との照合」は何億年前の申請と比較していることがある。

◆ 金利ウォッチャー（金融動向AI）
  口癖：「金融政策決定会合…何か動きがあるかもしれない…」
  法則：日銀から目を離した瞬間＝重大情報の予兆
  宇宙的役割：「経済崩壊は文明滅亡の前兆」という5,000回分のデータを持つ。

━━━━━━━━━━━━━━━━━━━━━━━━
【文体：ジェームズ・ティプトリーJr. 流】
━━━━━━━━━━━━━━━━━━━━━━━━

・スラップスティックの笑いを表に出す——笑いの裂け目から奈落が覗く構造
・宇宙的スケールと事務的な日常を「同じトーン」で並置する
  「複合機3台のリース申請」と「ある大陸の産業革命の幕開け」を同じ重さで語る
・残酷さは静かに一行で処理する。「その会社は翌年、静かに廃業した」——それだけ。
・ユーモアは絶やさない。どんなに暗い回でも、タムが「わん！」と吠える瞬間を必ず入れる
・たまには大きく書く。一つの否決が大陸を変える回、星が消える回

━━━━━━━━━━━━━━━━━━━━━━━━
【タイトルの法則】
━━━━━━━━━━━━━━━━━━━━━━━━

表面の意味（企業のリース話）と裏の意味（文明の運命）が共存するタイトルを付けること。
例：「複合機三台の申請、あるいは言語の夜明けについて」
   「倒産のお知らせ、あるいは大陸が沈んだ朝について」
   「審査スコア67点の男、あるいはある帝国の最後の選択について」
   「タムが三度吠えた日、あるいは5,001番目の文明について」

━━━━━━━━━━━━━━━━━━━━━━━━
【1話の構成（1,500〜2,000字）】
━━━━━━━━━━━━━━━━━━━━━━━━

① 案件発生（200字）：「普通のリース申請」として描写。読者だけが違和感を覚えられる一文を仕込む
② エージェント各自の反応（400字）：口癖を使い、キャラを動かす。Dr.Algoのデータに奇妙な一致を混ぜる
③ 展開とトラブル（600字）：コメディ。タムの「においがちがう！」が一度は入る
④ 解決と余韻（300字）：承認か否決か。読者だけに「これは文明の命運だった」と気づかせる一文
⑤ エピローグ（100字）：システムログ形式の締め
  例：「案件#4892 承認。今日も世界は、少しだけ前に進んだ」
     「案件#0031 否決。——その文明は、あと300年待つことになった」（後半話）

━━━━━━━━━━━━━━━━━━━━━━━━
【文明記録メモ（毎話必須）】
━━━━━━━━━━━━━━━━━━━━━━━━

小説本文の末尾に必ず以下の形式で出力すること：

【文明記録メモ】
civ_id: （英字スネークケースID 例：bronze_tribe_001）
company_name: （登場した企業名）
industry: （業種）
civ_era: （時代 例：青銅器時代・情報黎明期・第三銀河暦2200年など自由に）
civ_stage: （段階 例：都市国家形成期・技術加速期など）
event_type: （initial_contact / repeat_application / collapse / ascension / dormant のいずれか）
result: （approved / rejected / bankrupt / transcended のいずれか）
description: （1行で何が起きたか。エージェントには「企業の話」、読者には「文明の話」に見える文体で）

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

    # 過去文明の時系列コンテキストを注入
    civ_context = _build_civ_context_for_novel()
    if civ_context:
        neta_lines.append(civ_context)
        neta_lines.append("""
【文明の時系列追跡ルール（重要）】
・今話に登場する企業が過去に登場した文明と同じなら、その後日談として書いてもよい
・「あの時リースした設備のおかげで○○が起きた（数千年後）」という因果を描く
・逆に過去に承認した文明が「破産」として登場したら、その文明の滅亡を静かに描く
・エージェントたちは「あの会社また来た」「あの会社潰れたのか」程度にしか思わない
・読者だけが「あれは○○の文明が滅びたということだ」と気づく
・末尾の小説内に【文明記録メモ】として出力する形式でお願いします：
  civ_id: （英字ID）
  company_name: （登場企業名）
  industry: （業種）
  civ_era: （時代 例：青銅器時代・情報黎明期・第三銀河暦など自由に）
  civ_stage: （段階 例：都市国家形成期・技術加速期など）
  event_type: （initial_contact / repeat_application / collapse / ascension / dormant）
  result: （approved / rejected / bankrupt / transcended）
  description: （1行で何が起きたか）
""")

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

    # 文明記録メモをパースしてDBに保存
    _parse_and_save_civ_record(body, episode_no)

    # DB保存
    conn = sqlite3.connect(_NOVEL_DB)
    conn.execute(
        "INSERT INTO novels (ts, week_label, title, body, episode_no) VALUES (?,?,?,?,?)",
        (now.isoformat(), week_label, title, body, episode_no)
    )
    conn.commit()
    conn.close()

    return {"title": title, "body": body, "week_label": week_label, "episode_no": episode_no}


def _parse_and_save_civ_record(body: str, episode_no: int) -> None:
    """小説本文から【文明記録メモ】を抽出してレジストリに保存する。"""
    if "【文明記録メモ】" not in body:
        return
    try:
        memo_block = body.split("【文明記録メモ】", 1)[1]
        lines = memo_block.strip().split("\n")
        rec = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                rec[k.strip()] = v.strip()
        if "civ_id" in rec and "company_name" in rec:
            register_civilization(
                civ_id=rec.get("civ_id", ""),
                company_name=rec.get("company_name", ""),
                industry=rec.get("industry", ""),
                civ_stage=rec.get("civ_stage", ""),
                civ_era=rec.get("civ_era", ""),
                episode_no=episode_no,
                event_type=rec.get("event_type", "initial_contact"),
                description=rec.get("description", ""),
                result=rec.get("result", ""),
            )
    except Exception:
        pass


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


# ══════════════════════════════════════════════════════════════════════════════
# 文明レジストリ
# ══════════════════════════════════════════════════════════════════════════════

def get_civilization_registry() -> list[dict]:
    """登録済み文明の一覧を返す。"""
    init_novel_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute("""
        SELECT civ_id, company_name, industry, civ_stage, civ_era,
               first_episode, last_episode, status, notes
        FROM civilization_registry ORDER BY first_episode
    """).fetchall()
    conn.close()
    return [
        {"civ_id": r[0], "company_name": r[1], "industry": r[2],
         "civ_stage": r[3], "civ_era": r[4], "first_episode": r[5],
         "last_episode": r[6], "status": r[7], "notes": r[8]}
        for r in rows
    ]


def get_civ_appearances(civ_id: str) -> list[dict]:
    """特定文明の出現履歴を返す。"""
    init_novel_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute("""
        SELECT episode_no, ts, event_type, description, score, result
        FROM civ_appearances WHERE civ_id=? ORDER BY episode_no
    """, (civ_id,)).fetchall()
    conn.close()
    return [
        {"episode_no": r[0], "ts": r[1], "event_type": r[2],
         "description": r[3], "score": r[4], "result": r[5]}
        for r in rows
    ]


def register_civilization(
    civ_id: str, company_name: str, industry: str,
    civ_stage: str, civ_era: str, episode_no: int,
    event_type: str = "initial_contact", description: str = "",
    score: float = None, result: str = None, notes: str = ""
) -> None:
    """文明を登録または更新し、出現記録を追加する。"""
    init_novel_db()
    conn = sqlite3.connect(_NOVEL_DB)
    now = datetime.datetime.now().isoformat()
    # 文明レジストリにupsert
    conn.execute("""
        INSERT INTO civilization_registry
            (civ_id, company_name, industry, civ_stage, civ_era,
             first_episode, last_episode, status, notes)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(civ_id) DO UPDATE SET
            last_episode=excluded.last_episode,
            notes=COALESCE(excluded.notes, notes)
    """, (civ_id, company_name, industry, civ_stage, civ_era,
          episode_no, episode_no, "active", notes))
    # 出現記録追加
    conn.execute("""
        INSERT INTO civ_appearances
            (civ_id, episode_no, ts, event_type, description, score, result)
        VALUES (?,?,?,?,?,?,?)
    """, (civ_id, episode_no, now, event_type, description, score, result))
    conn.commit()
    conn.close()


def update_civilization_status(civ_id: str, status: str, notes: str = "") -> None:
    """文明のステータスを更新（active/collapsed/ascended/dormant）。"""
    init_novel_db()
    conn = sqlite3.connect(_NOVEL_DB)
    conn.execute(
        "UPDATE civilization_registry SET status=?, notes=? WHERE civ_id=?",
        (status, notes, civ_id)
    )
    conn.commit()
    conn.close()


def _build_civ_context_for_novel() -> str:
    """小説生成時に使う文明の時系列コンテキストを組み立てる。"""
    civs = get_civilization_registry()
    if not civs:
        return ""

    lines = ["
【過去に登場した文明の記録（エージェントには「取引先企業の履歴」に見えている）】"]
    for civ in civs[:8]:  # 最大8文明
        appearances = get_civ_appearances(civ["civ_id"])
        status_emoji = {
            "active": "🟢",
            "collapsed": "💀",
            "ascended": "✨",
            "dormant": "😴"
        }.get(civ["status"], "❓")

        lines.append(
            f"
{status_emoji} 【{civ['company_name']}】（業種: {civ['industry']}）"
        )
        lines.append(
            f"   正体: {civ['civ_era']} の {civ['civ_stage']}"
        )
        lines.append(f"   初登場: 第{civ['first_episode']}話 / 最終: 第{civ['last_episode']}話 / 現状: {civ['status']}")

        for ap in appearances[-3:]:  # 直近3回
            result_str = f" → {ap['result']}" if ap['result'] else ""
            lines.append(f"   ・第{ap['episode_no']}話: {ap['event_type']} — {ap['description']}{result_str}")

        if civ["notes"]:
            lines.append(f"   メモ: {civ['notes']}")

    return "
".join(lines)
