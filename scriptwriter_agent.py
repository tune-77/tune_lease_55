# -*- coding: utf-8 -*-
"""
scriptwriter_agent.py
=====================
脚本家AI（Scriptwriter Agent）モジュール。
ネットのRSSから最新の話題（IT・経済など）を取得し、
リース会社のAIエージェントたちが巻き込まれる「今週のカオスなプロット」を考案する。
考案されたプロットは json に保存され、小説家AI（Novelist）が読み込んで小説化する。
"""
from __future__ import annotations

import os
import json
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_PLOT_JSON  = os.path.join(_BASE_DIR, "data", "weekly_plot.json")
_MATH_DB    = os.path.join(_BASE_DIR, "data", "math_discoveries.db")
_NOVEL_DB   = os.path.join(_BASE_DIR, "data", "novel_records.db")

def get_recent_math_discoveries(limit: int = 2) -> list[dict]:
    """数学者エージェントが収集した最新の知見を取得する"""
    if not os.path.exists(_MATH_DB):
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(_MATH_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT method_name, summary, field_tag FROM math_discoveries ORDER BY ts DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

def get_past_civ_entities(limit: int = 5) -> list[str]:
    """過去の小説（文明年代記）に登場した企業や団体を取得する"""
    if not os.path.exists(_NOVEL_DB):
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(_NOVEL_DB)
        cursor = conn.execute("SELECT DISTINCT entity_name FROM civilization_registry ORDER BY ts DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []

def _backup_plot_before_write() -> None:
    """万が一の破損を防ぐためバックアップをとる"""
    if os.path.exists(_PLOT_JSON):
        bk_dir = os.path.join(_BASE_DIR, "data", "backups")
        os.makedirs(bk_dir, exist_ok=True)
        bk_path = os.path.join(bk_dir, f"weekly_plot.json.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        import shutil
        shutil.copy2(_PLOT_JSON, bk_path)


def fetch_internet_trends(num_items: int = 7) -> list[dict]:
    """
    Yahooニュース（IT・経済）などの公開RSSから最新トピックを取得する。
    Google News RSS などを利用してAPIキー無しで実現する。
    """
    # 検索キーワードをURLエンコード
    query = "AI OR テクノロジー OR 経済 OR ビジネス"
    encoded_query = urllib.parse.quote(query)
    # Google News RSS (日本語)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    
    news_items = []
    try:
        req = urllib.request.Request(
            rss_url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        # RSS2.0 の構造: <channel> -> <item> -> <title>, <link>, <pubDate>
        for item in root.findall('./channel/item'):
            title = item.findtext('title')
            link  = item.findtext('link')
            date_str = item.findtext('pubDate') or ""
            if title and link:
                # 「 - 〇〇新聞」などのサフィックスをなるべく除去してすっきりさせる
                clean_title = title.split(" - ")[0]
                news_items.append({
                    "title": clean_title,
                    "url": link,
                    "date": date_str
                })
            if len(news_items) >= num_items:
                break
    except Exception as e:
        print(f"[RSS Fetch Error] {e}")
        # フォールバック用のダミーニュース
        return [{"title": "最新のAIモデルが自我を持ち反乱か", "url": "", "date": "1時間前"}]
        
    return news_items


def generate_weekly_plot() -> dict:
    """
    LLMを呼び出し、取得したニュースから「今週のカオスなプロット」を生成する。
    """
    trends = fetch_internet_trends(5)
    maths  = get_recent_math_discoveries(2)
    civs   = get_past_civ_entities(5)
    
    trends_text = "\n".join([f"・{t['title']}" for t in trends])
    math_text   = "\n".join([f"・{m['method_name']}: {m['summary'][:100]}..." for m in maths]) if maths else "特になし"
    civ_text    = ", ".join(civs) if civs else "特になし"
    system_prompt = """
あなたは超一流の「AI脚本家」兼「審査戦略家」です。
最新のネットトレンド、高度な数学理論、そして過去の因縁（エンティティ）を融合させ、
今週の「AIエージェント・ドラマ」のプロットを作成してください。

また、そのプロットの文脈を活かして、実際のリース審査で担当者が使える「審査部を説得するためのキラーフレーズ」を3〜5個考案してください。
これらは、現在の社会情勢や最新理論を背景に、「なぜ今、この投資を承認すべきか」を熱く語る内容にしてください。

以下の形式のJSONでのみ答えてください。
{
  "title": "タイトル",
  "plot_text": "詳細なプロット本文（1000文字程度）",
  "story_arc": "物語のジャンル（サスペンス、コメディ等）",
  "killer_phrases": [
     {"text": "フレーズ本文", "reason": "そのフレーズを使う論理的根拠（数学理論やニュースとの関連）"}
  ]
}
"""
    
    user_prompt = (
        f"【最新ネット話題】\n{trends_text}\n\n"
        f"【Dr.Algoの最新発見（数式/理論）】\n{math_text}\n\n"
        f"【過去の因縁の企業・団体】\n{civ_text}\n\n"
        "これらの要素をすべて混ぜ合わせ、最高にカオスで知的なプロットを作成してください。"
    )
    
    import ai_chat
    import traceback
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        from components.agent_hub import _get_ai_settings
        engine, model, api_key, gemini_model = _get_ai_settings()
        
        # Force Gemini for stability during complex plot generation
        if gemini_model:
            engine = "gemini"
            model = gemini_model
        
        import components.agent_hub as ah
        import random as _rnd
        _analyzing_lines = [
            "最新ニュースとDr.Algoの理論を解析中... カオスな予感がするな。",
            "今週のネタを探している。経済ニュースが虐い...完璧なプロットの眼だ。",
            "ネットの話題とエージェントのデータをミキサーにかけている。今回はスパイスが效きそうだ。",
            "今週のトレンドをサーチ中。これをプロットに織り込むぞ。",
            "ニュースフィードを読みあさっている。ふむ、これとDr.Algoの研究を組み合わせれば...",
            "取材開始。今週のネタはなかなか辛辣だな。昇華させてやる。",
        ]
        ah._post_agent_thought("📝 脚本家AI", _rnd.choice(_analyzing_lines), "📝")
        
        resp_text = ai_chat._chat_for_thread(
            engine, model, 
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout_seconds=90,
            api_key=api_key,
            gemini_model=gemini_model
        )
        content = (resp_text.get("message") or {}).get("content", "")
        # Very Aggressive JSON Extraction
        import re
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_text = content[start_idx : end_idx + 1]
        else:
            clean_text = content.strip()

        try:
            result_json = json.loads(clean_text)
        except Exception:
            # フォールバック: LLMがJSONを返さなかった場合
            result_json = {
                "title": "AIの沈黙と、予期せぬ停滞",
                "plot_text": f"最新トピック（{trends[0].get('title') if trends else '...' }）の影響か、脚本家AIとの接続が一時的に途切れた。しかし、現場ではすでに何かが動き出している。Dr.Algoの最新手法が、誰も気づかないところで審査ロジックを書き換え始めたのだ...",
                "story_arc": "ミステリー"
            }
            import components.agent_hub as ah
            import random as _rnd2
            _err_lines = [
                "通信エラーが発生したが、蓄積されたデータから代替案を提示した。",
                "LLMとの回線が不安定だが、腕の見せどころだ。バックアッププロットで凌ぐ。",
                "通信障害は想定内。予備のネタ帳から代替プロットを組んだ。",
            ]
            ah._post_agent_thought("📝 脚本家AI", _rnd2.choice(_err_lines), "⚠️")
        
        plot_data = {
            "title": result_json.get("title", "無題のプロット"),
            "plot_text": result_json.get("plot_text", "プロットの生成に失敗しました。"),
            "story_arc": result_json.get("story_arc", "コメディ"),
            "killer_phrases": result_json.get("killer_phrases", []),
            "source_news": trends,
            "generated_at": now_str,
            "error": None
        }
        import components.agent_hub as ah
        import random as _rnd3
        _done_lines = [
            f"新プロット『{plot_data['title']}』を脱稿。ついでに審査部を黙らせるキラーフレーズも仕込んでおいたぜ。",
            f"『{plot_data['title']}』完成。今回のプロットは自信作だ。波乱丸も喜んでくれるだろう。",
            f"脚本完成。『{plot_data['title']}』——裏テーマは折り込み済み。気づくかな？",
            f"『{plot_data['title']}』を渡した。キラーフレーズもあるから審査部が使えるはずだ。",
            f"脱稿！『{plot_data['title']}』。これがボツになったら俺のせいだが、ヒットしたら全員のおかげだ。",
            f"『{plot_data['title']}』をデリバリー。今始めて行っておきます——感想は後で聞かせてくれ。",
        ]
        ah._post_agent_thought("📝 脚本家AI", _rnd3.choice(_done_lines), "🎯")
    except Exception as e:
        err_msg = str(e)
        print(f"[Plot Generation Error] {err_msg}")
        plot_data = {
            "title": "通信エラーによるプロット生成失敗",
            "plot_text": f"LLMとの通信に失敗しました。エラー詳細: {err_msg}",
            "story_arc": "エラー対応",
            "source_news": trends,
            "generated_at": now_str,
            "error": err_msg
        }

    # ファイルに保存
    _backup_plot_before_write()
    with open(_PLOT_JSON, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)
        
    return plot_data


def get_latest_plot() -> dict | None:
    """保存された最新のプロットを読み込む"""
    if not os.path.exists(_PLOT_JSON):
        return None
    try:
        with open(_PLOT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
