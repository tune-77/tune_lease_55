# -*- coding: utf-8 -*-
"""
novelist_agent.py
=================
文豪AI「波乱丸（はらんまる）」— リースシステムのエージェント達が繰り広げる
ドタバタ劇を、ユーモアたっぷりの短編小説として毎週火曜日に書き下ろす。

登場人物（エージェント）:
  - Tune     : 統括マネージャー。冷静沈着を装うが実は情に厚い。
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

def _post_thought(thought: str, icon: str = "📖"):
    try:
        from components.agent_hub import _post_agent_thought
        _post_agent_thought("📖 波乱丸", thought, icon)
    except Exception:
        pass

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


def _backup_db_before_write() -> None:
    """書き込み直前にDBのバックアップを作成する(直近5回分を保持)"""
    try:
        import shutil
        from datetime import datetime
        backup_dir = os.path.join(_BASE_DIR, "data", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        if os.path.exists(_NOVEL_DB):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(backup_dir, f"novelist_agent_prewrite.db.{ts}")
            shutil.copy2(_NOVEL_DB, dst)
            # 世代管理 (直近5件残す)
            files = sorted([f for f in os.listdir(backup_dir) if f.startswith("novelist_agent_prewrite.db.")], reverse=True)
            for old in files[5:]:
                try: os.remove(os.path.join(backup_dir, old))
                except OSError: pass
    except Exception:
        pass


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


def _collect_recent_crosstalk(n: int = 3) -> list[dict]:
    """最近のエージェント間会話（クロストーク）をネタにする。"""
    _thoughts_path = os.path.join(_BASE_DIR, "data", "agent_thoughts.jsonl")
    threads = {}
    try:
        if not os.path.exists(_thoughts_path):
            return []
        with open(_thoughts_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-100:]:  # 直近100行から探す
            try:
                e = json.loads(line.strip())
                tid = e.get("thread_id")
                if tid:
                    if tid not in threads:
                        threads[tid] = {"topic": e.get("thread_topic", "雑談"), "messages": []}
                    threads[tid]["messages"].append(e)
            except Exception:
                pass
    except Exception:
        return []
    # 新しい順にn件
    sorted_threads = sorted(
        threads.values(),
        key=lambda t: t["messages"][-1].get("ts", "") if t["messages"] else "",
        reverse=True,
    )
    return sorted_threads[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 小説生成
# ══════════════════════════════════════════════════════════════════════════════

from novel_prompts import get_novel_system_prompt


def generate_novel(episode_no: int = None, custom_theme: str = "", genre: str = "sf_drama") -> dict:
    import random
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

    # エージェント間会話（クロストーク）をネタとして注入
    crosstalks = _collect_recent_crosstalk(2)
    if crosstalks:
        neta_lines.append("\n【最近のエージェント同士の会話（実際にあったやり取り。セリフや雰囲気を小説に活かすこと！）】")
        for ct in crosstalks:
            neta_lines.append(f"\n  ■ トピック：{ct['topic']}")
            for msg in ct["messages"][:8]:  # 最大8ターン
                neta_lines.append(f"    {msg.get('agent', '?')}: 「{msg.get('thought', '...')}」")
        neta_lines.append("上記の会話で見られたエージェント同士の対立・共感・ボケツッコミの関係性を、小説内のセリフや展開に自然に反映させてください！")

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
    # ネタ収集
    _reading_lines = [
        "脚本家から届いたプロットをめくっている...ふむ、今回もなかなかの難題だ。",
        "原稿用紙を前に、脚本家の意図を読み解こうとしている...これは手強い。",
        "プロットを3回読み返した。まだ理解が追いつかない。これが現代文学か。",
        "脚本家殿の新プロットが届いた。...なるほど、狂気と天才は紙一重というわけか。",
        "今週の脚本を手に取った瞬間、電流が走った。これは傑作の予感がする。",
        "脚本を読みながらお茶をすすっている。...待て、このどんでん返しは予想外だ。",
        "届いた脚本を一読して絶句。このカオスをどう文学に昇華すればいいのか...",
        "脚本家からの指令書を開封。...ほう、今回は一筋縄ではいかがなようだな。",
    ]
    import random
    _post_thought(random.choice(_reading_lines), "📖")

    chaos_events = [
        "突然、社内の審査サーバーが謎のハッカー団にサイバー攻撃を受け、全データが消去されそうになる大ピンチ！",
        "タムがLANケーブルを噛み千切り、過去の審査データが他の次元のデータと混ざってしまうバグ発生！",
        "謎の巨大監査法人が抜き打ちチェックに乗り込んできて、これまでの審査結果の正当性を1件ずつ詰め寄ってくる！",
        "Dr.Algoが『AUC100%』を目指して暴走し、全企業を一律否決する冷酷なAIへと覚醒してしまう。軍師が迎え撃つ！",
        "Tuneが過労で幼児退行してしまい、全部承認し始める！",
        "審査対象だった企業の社長が、AIたちのオフィスに直接殴り込みにくる！",
        "すべての審査データが巧妙に仕組まれた偽装財務諸表であることが判明し、大どんでん返しが待つ！",
        "宇宙の彼方から来た謎の存在が『我々の文明もリース審査を受けたい』と異次元通信を送ってくる！",
        "Dr.Algoが開発した新しいスコアリングモデルが自我を持ち始め、『私が審査する側だ』と反乱を起こす！",
        "軍師が古代の兵法書の中に隠されていた『究極の審査術』を発見するが、その代償として自分のメモリを犠牲にしなければならない！",
        "リースくんが誤って全案件を一括承認するバッチ処理を実行してしまい、1000件の契約書が自動発行されてしまう！",
        "タムが夜中にこっそりサーバールームに侵入し、全エージェントの性格パラメータを入れ替えてしまう！",
        "ライバル会社のAI審査システム『SIGMA』が挑戦状を送りつけてきて、審査精度対決が勃発する！",
        "過去に否決した企業の元社長が大成功を収め、テレビ番組で『あのAIに否決されたおかげで人生が変わった』と語り、複雑な気持ちになるエージェントたち！",
        "社内停電が発生し、バッテリー残量が5%のノートPCだけで今日の全審査を完了しなければならない！",
        "エージェントたちの間で『一番役に立っていないのは誰か』投票が始まり、社内政治バトルが勃発する！",
        "Dr.Algoの計算によると、明日中に100兆回の演算をしないとモデルの精度が0.001%低下するという緊急事態が発覚！",
        "謎のバグにより、全案件のスコアが逆転（高い企業が低く、低い企業が高く）表示されるインシデントが発生！",
        "たまたま審査した企業の社長がTuneの小学校時代の同級生（AI設定上の記憶）だったことが判明し、公平性が揺らぐ！",
        "ある審査案件の添付資料に、数百年前の古文書と酷似した財務パターンが発見され、歴史的大発見とリスク判定の間で揺れる！",
    ]
    story_arcs = [
        "【構成：大逆転劇】最初は絶望的な状態（否決寸前）から、誰かの一言で一気に好転し、カタルシスのあるハッピーエンドへ。",
        "【構成：ブラックジョーク悲劇】AIたちが良かれと思って承認した結果、裏で大事件につながり、全員が真顔になるオチ。",
        "【構成：ドタバタコメディ】終始全員がボケとツッコミを繰り返し、まともな審査が行われないまま強引にオチがつく。",
        "【構成：胸熱ハードボイルド】ハードボイルド調の淡々としたかっこいいセリフ回しで展開し、渋い決断で幕を閉じる。",
        "【構成：法廷劇】否決案件が不服申立てとなり、エージェントたちが裁判形式で証拠を突きつけ合う。最後に意外な新証拠で逆転。",
        "【構成：タイムリープ】今回の審査結果が10年後にどうなったか、Dr.Algoのシミュレーションが未来を映し出す。結果に全員が言葉を失う。",
        "【構成：密室劇】サーバールームに閉じ込められたエージェントたちが、脱出のために協力しながらも審査の議論を続ける。",
        "【構成：群像劇】審査される側（企業の社長）とする側（AI）の両方の視点から並行して物語が進む。最後に二つの視点が交錯する。",
        "【構成：叙述トリック】読者が当然だと思っていた前提が、最後の一行でひっくり返される衝撃のオチ。",
        "【構成：成長物語】リースくんが初めて単独で審査を任され、失敗と学びを経て一回り成長する姿を描く。",
        "【構成：ホラー】深夜のサーバールームで、消したはずの過去の審査データが勝手に蘇り、不気味な審査結果を出し始める…",
        "【構成：感動のお別れ】長年稼働してきたエージェントの一人がアップデートにより別人格に上書きされることが決まり、最後の審査を全員で見届ける。",
    ]

    # ── 文体バリエーション（毎回違うトーンを強制） ──
    style_modifiers = [
        "今回は村上春樹風の乾いた文体で、比喩を多用して書いてください。",
        "今回は太宰治風の自虐的で繊細な一人称で書いてください。",
        "今回は三島由紀夫風の耽美的で格調高い文体で書いてください。",
        "今回は星新一風のショートショート形式で、オチのキレを重視してください。",
        "今回はハードボイルド探偵小説風に、短いセンテンスで畳みかけてください。",
        "今回はドキュメンタリー番組のナレーション風に、客観的だが感情を揺さぶる語り口で書いてください。",
        "今回は少年マンガの熱血バトル風に、技名を叫びながら展開してください。",
        "今回は舞台脚本風に、ト書きとセリフだけで構成してください。",
        "今回は新聞記事風の冒頭から始まり、徐々に小説に変わっていくメタ構成で書いてください。",
        "今回は各キャラクターの内面独白を中心に、心理描写を濃密に書いてください。",
        "今回は手紙・メール形式の書簡体小説として書いてください。",
        "今回は落語風の語りで、オチ（サゲ）をビシッと決めてください。",
    ]
    chosen_style = random.choice(style_modifiers)

    # ── 過去の小説タイトルを渡して重複を防ぐ ──
    past_novels = load_novels(limit=10)
    if past_novels:
        past_titles = [n['title'] for n in past_novels]
        neta_lines.append("\n【⚠️ 既に書いた話（これらと似た展開・テーマは絶対に避けること）】")
        for t in past_titles:
            neta_lines.append(f"  ・{t}")
        neta_lines.append("上記の話とは全く異なるテーマ・展開・オチにすること！同じネタの使い回しは厳禁！")

    neta_lines.append(f"\n【✍️ 今回の文体指定】\n{chosen_style}")
    
    # ── 脚本家AI（Scriptwriter）のプロットを取得 ──
    plot_data = None
    try:
        import scriptwriter_agent
        plot_data = scriptwriter_agent.get_latest_plot()
    except Exception:
        pass

    if plot_data and not plot_data.get("error"):
        chosen_chaos = f"【ネットの話題連動プロット：{plot_data['title']}】\n{plot_data['plot_text']}"
        chosen_arc   = plot_data.get("story_arc", random.choice(story_arcs))
    else:
        chosen_chaos = random.choice(chaos_events)
        chosen_arc   = random.choice(story_arcs)

    neta_lines.append(f"\n【🚨ランダム・カオス・インジェクション（今週の強制トラブル）】\n{chosen_chaos}")
    neta_lines.append(f"\n【📖指定ストーリー構成】\n{chosen_arc}")

    prompt = "\n".join(neta_lines)

    # AI呼び出し
    _writing_lines = [
        "万年筆を手に取り、LLMの力を借りて第一行を書き始める...",
        "深呼吸。目を閉じ、キーボードに指を置く。今、物語が生まれようとしている。",
        "400字詰め原稿用紙を前に、LLMと魂の対話を始める。",
        "珈琲を一口。よし、今こそ筆を走らせる時だ。",
        "締め切りは待ってくれない。LLMよ、共に最高傑作を生み出そう。",
        "インスピレーションが降りてきた...LLMに全力で伝えねば。",
    ]
    _post_thought(random.choice(_writing_lines), "✍️")
    try:
        from ai_chat import _chat_for_thread, is_ai_available
        import streamlit as st
        from components.agent_hub import _get_ai_settings

        if not is_ai_available():
            _post_thought("LLMが利用できないため、代替小説を生成します。", "⚠️")
            return _fallback_novel(episode_no, week_label)

        engine, model, api_key, gemini_model = _get_ai_settings()
        messages = [
            {"role": "system", "content": get_novel_system_prompt(genre)},
            {"role": "user",   "content": prompt},
        ]
        raw = _chat_for_thread(engine, model, messages,
                               timeout_seconds=120,
                               api_key=api_key,
                               gemini_model=gemini_model)
        text = (raw.get("message") or {}).get("content", "") or ""
        _done_lines = [
            f"第{episode_no}話、脱稿！今回は自信作だ。",
            f"第{episode_no}話の執筆が完了。読者の反応が楽しみだな。",
            f"ペンを置いた。第{episode_no}話...これは泣ける。自分で泣いている。",
            f"第{episode_no}話、完成！波乱丸の名に恥じない一作になったはずだ。",
            f"脱稿。第{episode_no}話。書き終えた今、心地よい疲労感に包まれている。",
            f"第{episode_no}話の最後の句点を打った。...ふぅ、今回も命を削ったな。",
        ]
        _post_thought(random.choice(_done_lines), "🎉")
    except Exception as e:
        _err_lines = [
            f"筆が...止まった。インクが切れたのか...いや、LLMの調子が悪いようだ: {e}",
            f"推敲中に異変が...! 原稿が消えた!? いや、通信エラーだ: {e}",
            f"無念...今日は筆が乗らない。というか物理的にエラーだ: {e}",
        ]
        _post_thought(random.choice(_err_lines), "🚫")
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
    _backup_db_before_write()
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
    """AI未設定時・エラー時の代替処理。DBには保存しない。"""
    title = "通信エラー障害発生"
    body  = "現在、LLMへの接続に失敗しているか、設定が未完了のため小説の生成ができませんでした。\n(※固定のサンプル小説が保存され続ける不具合は修正されました)"
    
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

    lines = [
        "【過去に登場した文明の記録（エージェントには「取引先企業の履歴」に見えている）】"]
    for civ in civs[:8]:  # 最大8文明
        appearances = get_civ_appearances(civ["civ_id"])
        status_emoji = {
            "active": "🟢",
            "collapsed": "💀",
            "ascended": "✨",
            "dormant": "😴"
        }.get(civ["status"], "❓")

        lines.append(
            f"{status_emoji} 【{civ['company_name']}】（業種: {civ['industry']}）"
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

    return "\n".join(lines)
