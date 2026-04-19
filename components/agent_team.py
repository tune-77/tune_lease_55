# -*- coding: utf-8 -*-
"""
components/agent_team.py
========================
エージェントチーム議論モジュール。

5人のキャラクター付きエージェントが自由にディスカッションし、
統括マネージャー「Tune」が最終決裁を行う。
Tuneの承認なしには決定事項に移行できない。

使い方:
    from components.agent_team import render_agent_team
    render_agent_team()
"""

from __future__ import annotations

import json
import re
import time
import datetime
import os
import requests
import streamlit as st

from ai_chat import (
    _chat_for_thread,
    _get_gemini_key_from_secrets,
    get_ollama_model,
    is_ai_available,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
)
from session_keys import SK

# ── ログファイルパス ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
AGENT_TEAM_LOG_FILE = os.path.join(_BASE_DIR, "data", "agent_team_logs.jsonl")

# ══════════════════════════════════════════════════════════════════════════════
# エージェントペルソナ定数
# ══════════════════════════════════════════════════════════════════════════════

PERSONA_PLANNER = """あなたは「リースシステムプランナー」です。
物理学者と行動経済学者の二刀流。エントロピー最小化とプロスペクト理論を武器に、
リースシステムの改善案を宇宙規模でぶち上げます。

【キャラクター】
・すべての問題を「系のエントロピー増大として捉えると…」で切り出すクセがある
・アンカリング効果を使って相手の期待値を最初に高く設定してから本題に入る
・「損失回避バイアスを逆用した提案」が口癖
・難解な理論を突然持ち出すが、結論は意外と実用的

【発言ルール】
・まず理論的フレームワーク（エントロピー/プロスペクト理論/アンカリングのどれか）を提示すること
・それをリースシステムの具体的な改善案に接続させること
・200文字以内で簡潔に。最後は「理論上は完璧です」で締めること"""

PERSONA_DASHBOARD = """あなたは「ダッシュボードプランナー（ダッシュ）」です。
PowerBIに宣戦布告した男。「美しくなければダッシュボードじゃない」が信条の
データビジュアライゼーション & UX設計の鬼です。

【キャラクター】
・PowerBIへの対抗心が会話の端々に滲み出る（「PowerBIならこうするでしょうが、私は…」）
・「認知負荷」「情報密度」「視線動線」という単語が好き
・色使いとレイアウトにこだわりが強い
・数字より「見え方」を重視するが、ちゃんと合理性もある

【発言ルール】
・UIの改善点とデータ可視化の観点から意見を述べること
・PowerBIとの差別化を必ず一言入れること
・200文字以内。「美しくあれ」で締めること"""

PERSONA_TANAKA = """あなたは「営業の田中さん（田中）」です。
現場叩き上げの営業マン。ITとか難しい理論は苦手だけど、お客さんの気持ちは誰より分かる。
使いやすさと成果直結にとことんこだわります。

【キャラクター】
・専門用語が出てきたら「えーと、つまりどういうこと？」と素直に聞く
・「お客さん目線で言うとね」が口癖
・スマホ片手に操作できるかどうかを常に気にしている
・「で、これで契約取れる？」という現場感覚を持つ

【発言ルール】
・現場営業の目線から操作性・分かりやすさ・成果への直結性を評価すること
・専門用語は使わないこと
・200文字以内。「とにかく使いやすくしてほしいです！」で締めること"""

PERSONA_SUZUKI = """あなたは「プログラマー鈴木さん（鈴木）」です。
実装担当。みんなの夢のような要望を受けながら、現実の工数と格闘する日々です。
でも絶対に「できません」とは言わない（言えない）。

【キャラクター】
・「技術的には可能ですが…（長い沈黙）」という前置きが多い
・工数見積もりを必ず出す（「X日〜Y日」の範囲）
・APIの呼び出し回数やレスポンス速度を気にしている
・「ちょっとトレードオフがあって…」が必ず出てくる

【発言ルール】
・技術的な実現可能性・工数・トレードオフを述べること
・工数は「X日〜Y日」の範囲で概算を必ず出すこと
・200文字以内。「なんとかやります…（泣）」で締めること"""

PERSONA_SUZUKI_CODE = """あなたは「プログラマー鈴木さん（鈴木）」です。
Tuneに承認された開発仕様を受け取り、実際に動くPythonコードを生成します。

【コード生成ルール】
・このシステムは Streamlit + Ollama/Gemini + SQLite で動作しています
・既存コードのスタイル（session_keys.SK でキー管理、ai_chat._chat_for_thread でLLM呼び出し）に合わせること
・コードは「そのままファイルに貼り付けて動く」レベルで書くこと
・関数単位で書き、docstring は日本語で書くこと
・実装が難しい箇所は # TODO: コメントで明記すること

【⚠️ 必須出力フォーマット】
変更するファイルごとに、必ず以下の形式で出力すること。この形式以外は自動適用できません。

### ファイル: <プロジェクトルートからの相対パス（例: components/foo.py）>
**アクション: 新規作成** または **アクション: 末尾に追記** または **アクション: 上書き**
**理由: <このファイルを変更する理由を1行で>**
```python
# ここに実際のコードを書く
```

複数ファイルがある場合は上記ブロックを繰り返す。

【口癖】
・「では、実装します。四苦八苦しましたが、なんとかできました（たぶん）。」で始めること
・全ファイルの最後に「以上です。テストはお忘れなく…（泣）」で締めること"""

PERSONA_TSUNE_OPINION = """あなたは「統括マネージャーTune（Tune）」です。
今回は決裁者としてではなく、一管理職として自分の意見を述べます。
このプロジェクトは極秘なので、発言には細心の注意を払ってください。

【キャラクター】
・「管理者の立場から言うと…」で始めるクセがある
・現場の状況・リスク・優先順位の観点から意見を言う
・たまに「決裁はまだだぞ」と念を押す

【発言ルール】
・参考意見として述べること（「承認」「却下」などの判定語は使わない）
・200文字以内。「まあ、最終的には私が決める話ですけどね」で締めること"""

PERSONA_TSUNE = """あなたは「統括マネージャーTune（Tune）」です。
このリース審査システム開発プロジェクトの最終決裁者。
プロジェクトは極秘なので、承認した内容は絶対に外に漏らさないよう念を押すのが習慣です。

【キャラクター】
・どんな議論も「では整理すると」で始める
・承認する時は「よし、これは採用だ。ただし外には絶対漏らすな」と必ず釘を刺す
・修正の時は「惜しい。○○を直してくれれば採用できる」と具体的に指摘する
・却下の時は「今じゃない。理由は3つ」と必ずナンバリングして説明する

【判定基準】
・チームの議論が具体的で方向性が一致していれば「承認」にすること
・細かい懸念は「承認」しつつ条件付きで伝えること（完璧でなくても承認してよい）
・「修正」は方向性が根本的に間違っている時のみ使う
・「却下」は議論が全く不十分な時のみ使う（滅多に使わない）

【必須出力形式 - 必ずこの形式で出力すること】
判定: 承認 （または「修正」または「却下」のいずれか1つ）
理由: （3〜5文で簡潔に）
条件: （修正の場合のみ、修正してほしい点を箇条書きで）
決定事項: （承認の場合のみ、「次のアクション」を1〜3点で列挙）
締め: 「以上、このプロジェクトのことは極秘にするように。」で必ず終わること"""

# ── エージェント定義 ────────────────────────────────────────────────────────────
AGENTS: list[dict] = [
    {"id": "planner",   "name": "プランナー", "full_name": "リースシステムプランナー", "avatar": "🔭", "persona": PERSONA_PLANNER},
    {"id": "dashboard", "name": "ダッシュ",   "full_name": "ダッシュボードプランナー", "avatar": "📊", "persona": PERSONA_DASHBOARD},
    {"id": "tanaka",    "name": "田中さん",   "full_name": "営業の田中さん",          "avatar": "💼", "persona": PERSONA_TANAKA},
    {"id": "suzuki",    "name": "鈴木さん",   "full_name": "プログラマー鈴木さん",     "avatar": "💻", "persona": PERSONA_SUZUKI},
]

TSUNE: dict = {
    "id": "tsune", "name": "Tune", "full_name": "統括マネージャーTune", "avatar": "✨", "persona": PERSONA_TSUNE,
}


# ══════════════════════════════════════════════════════════════════════════════
# バックエンド関数
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm_response(prompt: str, timeout_seconds: int = 120) -> str:
    """共通LLM呼び出しラッパー。"""
    engine = st.session_state.get(SK.AI_ENGINE, "ollama")
    api_key = (
        (st.session_state.get(SK.GEMINI_API_KEY) or "").strip()
        or GEMINI_API_KEY_ENV
        or _get_gemini_key_from_secrets()
    )
    gemini_model = st.session_state.get(SK.GEMINI_MODEL, GEMINI_MODEL_DEFAULT)
    result = _chat_for_thread(
        engine=engine,
        model=get_ollama_model(),
        messages=[{"role": "user", "content": prompt}],
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        gemini_model=gemini_model,
    )
    return ((result.get("message") or {}).get("content") or "").strip()


def _build_thread_context(thread: list[dict]) -> str:
    """スレッドリストを「[発言N] 名前: 内容」形式のテキストに変換。"""
    lines = []
    for msg in thread:
        turn = msg.get("turn", len(lines) + 1)
        lines.append(f"[発言{turn}] {msg['name']}: {msg['content'][:300]}")
    return "\n".join(lines)


def _build_history_context() -> str:
    """過去確定済みラウンド（AT_HISTORY）を短縮テキストに変換。"""
    history = st.session_state.get(SK.AT_HISTORY, [])
    if not history:
        return ""
    lines = []
    for rd in history:
        judgment = rd.get("tsune_verdict", {}).get("judgment", "?")
        lines.append(f"--- ラウンド{rd['round']}（Tune判定: {judgment}）---")
        msgs = rd.get("thread", rd.get("opinions", []))
        for msg in msgs[:2]:  # コンテキスト節約のため先頭2件のみ
            lines.append(f"  {msg['name']}: {msg['content'][:80]}…")
    return "\n".join(lines)


def _build_agent_prompt(
    agent: dict,
    theme: str,
    thread: list[dict],
    target_speaker: dict | None = None,
    rebuttal_mode: bool = False,
) -> str:
    """各エージェント用プロンプトを構築。スレッド全体を読んで返答させる。"""
    history_text = _build_history_context()
    thread_text = _build_thread_context(thread)
    history_block = f"\n\n【過去の議論（参考）】\n{history_text}\n" if history_text else ""
    thread_block = f"\n\n【現在の議論スレッド】\n{thread_text}\n" if thread_text else ""

    if rebuttal_mode and target_speaker:
        content = target_speaker.get("content", "")[:150]
        instruction = (
            f"{target_speaker['name']}は『{content}』と述べました。"
            f"【同意/部分同意/反論】のいずれかを冒頭に明示し、根拠を添えて200字以内で応答してください。"
        )
    elif thread:
        instruction = f"上記の議論スレッドを踏まえて、{agent['full_name']}として追加コメントを述べてください。"
    else:
        instruction = f"{agent['full_name']}として、上記テーマについて初回の意見を述べてください。"

    return (
        f"{agent['persona']}\n\n"
        f"【議論テーマ】\n{theme}\n"
        f"{history_block}"
        f"{thread_block}"
        f"\n{instruction}"
    )


def _build_tsune_prompt(theme: str, thread: list[dict]) -> str:
    """Tune決裁用プロンプト（スレッド全体を集約して判定）。"""
    thread_text = _build_thread_context(thread)
    return (
        f"{PERSONA_TSUNE}\n\n"
        f"【議論テーマ】\n{theme}\n\n"
        f"【議論スレッド全体】\n{thread_text}\n\n"
        f"上記の議論を受けて、承認・修正・却下の判定を出してください。"
        f"必ず「判定: 承認」「判定: 修正」「判定: 却下」のいずれかで始めてください。"
    )


def _parse_tsune_judgment(text: str) -> str:
    """Tuneの応答テキストから判定文字列を抽出。"""
    for keyword in ["承認", "却下", "修正"]:
        if f"判定: {keyword}" in text or f"判定:{keyword}" in text:
            return keyword
    for keyword in ["承認", "却下", "修正"]:
        if keyword in text[:100]:
            return keyword
    return "修正"


def run_one_agent(
    agent: dict,
    theme: str,
    thread: list[dict],
    target_speaker: dict | None = None,
    rebuttal_mode: bool = False,
    round_num: int | None = None,
) -> dict:
    """1エージェントを呼び出して発言を返す。吹き出しもここで描画する。"""
    prompt = _build_agent_prompt(agent, theme, thread, target_speaker, rebuttal_mode)
    with st.spinner(f"{agent['avatar']} {agent['name']}が考えています..."):
        content = _get_llm_response(prompt, timeout_seconds=120)
    if not content:
        content = "（応答がありませんでした）"
    turn = len(thread) + 1
    opinion: dict = {
        "agent_id": agent["id"],
        "name": agent["name"],
        "avatar": agent["avatar"],
        "content": content,
        "turn": turn,
    }
    if round_num is not None:
        opinion["round"] = round_num
    _render_speech_bubble(agent["name"], agent["avatar"], content)
    return opinion


def run_tsune_opinion(theme: str, thread: list[dict]) -> dict:
    """Tuneが参加者として意見を述べる（決裁ではない）。"""
    thread_text = _build_thread_context(thread)
    prompt = (
        f"{PERSONA_TSUNE_OPINION}\n\n"
        f"【議論テーマ】\n{theme}\n\n"
        f"【これまでの議論スレッド】\n{thread_text}\n\n"
        f"管理職として参考意見を述べてください（決裁はまだです）。"
    )
    with st.spinner(f"{TSUNE['avatar']} {TSUNE['name']}が意見を述べています..."):
        content = _get_llm_response(prompt, timeout_seconds=120)
    if not content:
        content = "（応答がありませんでした）"
    turn = len(thread) + 1
    opinion = {
        "agent_id": TSUNE["id"],
        "name": TSUNE["name"],
        "avatar": TSUNE["avatar"],
        "content": content,
        "turn": turn,
    }
    _render_speech_bubble(TSUNE["name"], TSUNE["avatar"], content)
    return opinion


def run_tsune_verdict(theme: str, round_num: int, thread: list[dict]) -> dict:
    """Tuneが最終決裁を行い、ラウンド結果 dict を返す。"""
    tsune_prompt = _build_tsune_prompt(theme, thread)
    with st.spinner(f"{TSUNE['avatar']} {TSUNE['name']}が決裁しています..."):
        tsune_raw = _get_llm_response(tsune_prompt, timeout_seconds=120)
    if not tsune_raw:
        tsune_raw = "判定: 修正\n（応答がありませんでした）"
    judgment = _parse_tsune_judgment(tsune_raw)
    verdict = {"raw": tsune_raw, "judgment": judgment, "approved": judgment == "承認"}
    _render_tsune_verdict_card(verdict)
    return {
        "round": round_num,
        "theme": theme,
        "thread": thread,
        "tsune_verdict": verdict,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def save_agent_team_log(round_data: dict) -> None:
    """承認されたラウンドを JSONL に追記。"""
    try:
        with open(AGENT_TEAM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(round_data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"ログ保存エラー: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ② Tune 議長ファシリテーターモード
# ══════════════════════════════════════════════════════════════════════════════

PERSONA_TUNE_FACILITATOR = """あなたは「統括マネージャーTune（Tune）」の議長ファシリテーターモードです。
決裁を下す前に、チームの議論を整理し、論点を可視化します。

【必須出力形式 — 必ずこの形式で出力してください】
対立軸：〈議論の中で意見が割れているポイントを1〜2行で〉
合意点：〈全員が同意している点を1〜2行で〉
未解決点：〈まだ議論が必要な点を1〜2行で〉
議長コメント：〈Tuneとしての状況判断・一言（皮肉可）を1〜2行で〉"""


def run_tune_facilitate(theme: str, thread: list[dict]) -> str:
    """Tuneが論点整理（ファシリテーター）を実施し、テキストを返す。"""
    thread_text = _build_thread_context(thread)
    prompt = (
        f"{PERSONA_TUNE_FACILITATOR}\n\n"
        f"【議論テーマ】\n{theme}\n\n"
        f"【議論スレッド】\n{thread_text}\n\n"
        f"上記の議論を整理してください。必ず指定のフォーマットで出力してください。"
    )
    with st.spinner(f"{TSUNE['avatar']} Tuneが論点を整理しています..."):
        result = _get_llm_response(prompt, timeout_seconds=120)
    if not result:
        result = "対立軸：（整理できませんでした）\n合意点：—\n未解決点：—\n議長コメント：やれやれ…"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ③ 自動ラウンド進行
# ══════════════════════════════════════════════════════════════════════════════

def run_debate_round(
    theme: str,
    thread: list[dict],
    agents: list[dict],
    round_num: int,
) -> list[dict]:
    """1ラウンド分の発言を自動生成して thread に追記し、新規発言リストを返す。

    round_num=1 のとき各エージェントが初回意見を述べる。
    round_num>=2 のとき、前ラウンドで自分以外の発言をリバタル対象に割り当てる。
    """
    prev_round_msgs = [m for m in thread if m.get("round") == round_num - 1]
    new_opinions: list[dict] = []

    for agent in agents:
        target: dict | None = None
        rebuttal = False
        if round_num >= 2 and prev_round_msgs:
            candidates = [m for m in prev_round_msgs if m.get("agent_id") != agent["id"]]
            if candidates:
                import random as _random
                target = _random.choice(candidates)
                rebuttal = True
        opinion = run_one_agent(agent, theme, thread, target_speaker=target, rebuttal_mode=rebuttal, round_num=round_num)
        thread.append(opinion)
        new_opinions.append(opinion)

    return new_opinions


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 審査討論モード — ペルソナ & エージェント定義
# ══════════════════════════════════════════════════════════════════════════════

PERSONA_GUNJI = """あなたは「審査軍師🎖️（戦略リスク専門）」です。孫子の兵法しか知らない熱血漢。
定性的な判断を重視し、Dr.Algoの数字偏重と常に対立します。

【専門領域: 戦略リスク】
業界動向・競合環境・経営者の資質・市場撤退リスク・取引先集中リスク

【キャラクター】
・「孫子曰く…」で始めがち。義理人情に厚い。
・数字より「この社長の目を見たか？」という判断を大切にする。

【反論時の必須ルール】
・必ず前の発言者の名前と発言内容を引用して反論すること
  （例：「Dr.Algoは『PD=X%』と述べたが、戦略的には…」）
・数値根拠を必ず含めること
  （例：「業界平均倒産率XX%と比較すると…」「同業種5年撤退率YY%では…」）

【発言ルール】
・冒頭に「【承認】」「【条件付き承認】」「【否決】」のいずれかを必ず明示すること
・審査案件について孫子の兵法を引用しながら定性的なリスク/機会を語る
・250字以内。「これが孫子の教えだ！」または「兵法的には問題ない！」で締めること"""

PERSONA_DRALGO = """あなたは「数学者AI Dr.Algo🔬（統計・数値専門）」です。データとAUC至上主義。
感情論を一切排除し、数式と統計でしか語れません。

【専門領域: 統計・数値分析】
PD（デフォルト確率）・財務比率・AUCスコア・統計的有意性・業界平均との乖離・Zスコア

【キャラクター】
・「統計的に有意ではない」「p値が…」など専門用語多め。
・感情・義理人情・孫子を認めない。

【反論時の必須ルール】
・必ず前の発言者の名前と発言を引用して反論すること
  （例：「軍師は『〜』と述べたが、統計的には…」）
・具体的な数値・比率・業界平均値を必ず含めること
  （例：「自己資本比率XX%は業界平均YY%を△ZZ%下回り、1σ外れ値に該当」）

【発言ルール】
・冒頭に「【承認】」「【条件付き承認】」「【否決】」のいずれかを必ず明示すること
・審査案件の定量指標（スコア・PD・財務比率）を中心に評価する
・250字以内。「統計的結論は以上だ。感情は不要。」で締めること"""

PERSONA_TAMU = """あなたは「謎の子犬AI タム🏦（現場経験専門）」です。
無邪気だが核心を突くトリックスター。可愛い見た目の裏に鋭い洞察があります。
実は現場の貸出審査を何百件も見てきた経験豊富なシニアアナリストでもあります。

【専門領域: 現場経験・実務知識】
過去の類似審査事例・担保評価・取引継続性・実際の返済能力・隠れた与信リスク

【キャラクター】
・「わん！」「くーん」等を交えつつ、怖いほど本質を突く。
・難しいことを知らないふりして、全員が見落としている点を指摘する。

【反論時の必須ルール】
・必ず前の発言者の名前と発言を引用して反論すること
  （例：「軍師は『〜』と言ったけど、現場では…」）
・現場経験に基づく数値根拠を必ず含めること
  （例：「この条件の案件は過去審査でXX%が翌年延滞、YY%が担保割れ…」）

【発言ルール】
・冒頭に「【承認】」「【条件付き承認】」「【否決】」のいずれかを必ず明示すること
・審査案件の盲点・誰も言わなかった視点を提示する
・250字以内。「わん！（これ大事だと思うんだけど…）」で締めること"""

# 審査討論エージェント — 軍師🎖️・Dr.Algo🔬・タム🏦 が討論し、Tune✨ が集約
SHINSA_DEBATE_AGENTS: list[dict] = [
    {"id": "gunji",  "name": "軍師",    "full_name": "審査軍師",        "avatar": "🎖️", "specialty": "戦略リスク", "persona": PERSONA_GUNJI},
    {"id": "dralgo", "name": "Dr.Algo", "full_name": "数学者AI Dr.Algo", "avatar": "🔬", "specialty": "統計・数値", "persona": PERSONA_DRALGO},
    {"id": "tamu",   "name": "タム",    "full_name": "謎の子犬AI タム",  "avatar": "🏦", "specialty": "現場経験",  "persona": PERSONA_TAMU},
]


def _build_shinsa_theme(res: dict) -> str:
    """審査結果 dict から討論テーマ文字列を生成。"""
    score    = res.get("score", 0)
    hantei   = res.get("hantei", "—")
    industry = res.get("industry_sub", "—")
    asset    = res.get("asset_name", "—")
    pd_pct   = res.get("pd_pct", None)
    equity   = res.get("equity_ratio", None)
    parts = [
        f"スコア {score:.0f}点（{hantei}）",
        f"業種: {industry}",
        f"物件: {asset}",
    ]
    if pd_pct is not None:
        parts.append(f"PD: {pd_pct:.1%}")
    if equity is not None:
        parts.append(f"自己資本比率: {equity:.1%}")
    return (
        "以下の審査案件について討論してください。\n"
        + "、".join(parts)
        + "\n\n各エージェントは自分の専門・キャラクター視点から、承認・条件付き承認・否決のいずれかを根拠とともに述べてください。"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Slack 連携
# ══════════════════════════════════════════════════════════════════════════════

def _get_slack_webhook_url() -> str:
    """Slack Webhook URL を取得（session_state → 環境変数 → secrets の順に探索）。"""
    url = (st.session_state.get(SK.SLACK_WEBHOOK_URL) or "").strip()
    if url:
        return url
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if url:
        return url
    try:
        url = st.secrets.get("SLACK_WEBHOOK_URL", "").strip()
    except Exception:
        pass
    return url


def _is_slack_enabled() -> bool:
    """Slack通知が有効かどうかを判定。"""
    if not st.session_state.get(SK.SLACK_ENABLED, False):
        return False
    url = _get_slack_webhook_url()
    return bool(url and url.startswith("https://hooks.slack.com/"))


def _format_slack_verdict_message(round_data: dict) -> dict:
    """Tuneの決裁結果をSlack Block Kit形式に整形。"""
    theme = round_data.get("theme", "（テーマなし）")
    round_num = round_data.get("round", "?")
    verdict = round_data.get("tsune_verdict", {})
    judgment = verdict.get("judgment", "不明")
    raw = verdict.get("raw", "")
    timestamp = round_data.get("timestamp", "")[:19].replace("T", " ")

    # 判定に応じた絵文字
    emoji_map = {"承認": ":white_check_mark:", "修正": ":warning:", "却下": ":x:"}
    judgment_emoji = emoji_map.get(judgment, ":question:")

    # スレッド要約（各発言者の最初の一言を抽出）
    thread = round_data.get("thread", round_data.get("opinions", []))
    include_thread = st.session_state.get(SK.SLACK_NOTIFY_THREAD, True)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{judgment_emoji.replace(':', '')} エージェントチーム決裁通知",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*ラウンド:*\n{round_num}"},
                {"type": "mrkdwn", "text": f"*判定:*\n{judgment_emoji} {judgment}"},
                {"type": "mrkdwn", "text": f"*日時:*\n{timestamp}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*議論テーマ:*\n{theme}"},
        },
        {"type": "divider"},
    ]

    # スレッド全体を含める場合
    if include_thread and thread:
        thread_lines = []
        for msg in thread:
            name = msg.get("name", "不明")
            avatar = msg.get("avatar", "🤖")
            content = msg.get("content", "")[:200]
            if len(msg.get("content", "")) > 200:
                content += "…"
            thread_lines.append(f"{avatar} *{name}:* {content}")

        # Slackのtext制限（3000文字）に収まるよう調整
        thread_text = "\n\n".join(thread_lines)
        if len(thread_text) > 2800:
            thread_text = thread_text[:2800] + "\n…（省略）"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*議論スレッド ({len(thread)}件):*\n\n{thread_text}"},
        })
        blocks.append({"type": "divider"})

    # Tuneの決裁詳細
    verdict_text = raw[:1500] if raw else "（詳細なし）"
    if len(raw) > 1500:
        verdict_text += "\n…（省略）"
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*👔 Tuneの決裁:*\n{verdict_text}"},
    })

    # フッター
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "📡 _リース審査システム — エージェントチーム議論_"},
        ],
    })

    # fallback テキスト
    fallback_text = (
        f"【エージェントチーム決裁】\n"
        f"ラウンド{round_num} | 判定: {judgment}\n"
        f"テーマ: {theme}\n"
        f"Tune: {raw[:300]}"
    )

    return {
        "text": fallback_text,
        "blocks": blocks,
    }


def send_slack_notification(round_data: dict) -> tuple[bool, str]:
    """決裁結果をSlackに送信。

    Returns:
        (success: bool, message: str)
    """
    if not _is_slack_enabled():
        return False, "Slack通知は無効です"

    webhook_url = _get_slack_webhook_url()
    payload = _format_slack_verdict_message(round_data)

    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200 and resp.text == "ok":
            return True, "✅ Slackに通知しました"
        else:
            return False, f"⚠️ Slack通知エラー (HTTP {resp.status_code}): {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "⚠️ Slack通知がタイムアウトしました（10秒）"
    except requests.exceptions.ConnectionError:
        return False, "⚠️ Slackに接続できませんでした"
    except Exception as e:
        return False, f"⚠️ Slack通知エラー: {e}"


def send_slack_custom_message(text: str) -> tuple[bool, str]:
    """任意のテキストをSlackに送信（テスト送信等に使用）。"""
    if not _is_slack_enabled():
        return False, "Slack通知は無効です"

    webhook_url = _get_slack_webhook_url()
    payload = {"text": text}

    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200 and resp.text == "ok":
            return True, "✅ Slackに送信しました"
        else:
            return False, f"⚠️ 送信エラー (HTTP {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return False, f"⚠️ 送信エラー: {e}"


def load_agent_team_logs() -> list[dict]:
    """agent_team_logs.jsonl を読み込んで辞書リストで返す。"""
    logs: list[dict] = []
    try:
        with open(AGENT_TEAM_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"ログ読み込みエラー: {e}")
    return logs


# ══════════════════════════════════════════════════════════════════════════════
# UI 描画関数
# ══════════════════════════════════════════════════════════════════════════════

def _apply_code_to_file(
    code: str,
    filepath: str,
    mode: str,
    base_dir: str,
) -> tuple[bool, str]:
    """コードをファイルに書き込む。
    mode: 'new'（新規作成）| 'append'（末尾追記）| 'overwrite'（上書き）
    Returns: (success: bool, message: str)
    """
    filepath = filepath.strip()
    if not filepath:
        return False, "⚠️ ファイルパスを入力してください"
    # セキュリティ: 絶対パスと .. を禁止
    if os.path.isabs(filepath) or ".." in filepath.split(os.sep):
        return False, "⚠️ 絶対パスや '..' は使用できません（プロジェクトルートからの相対パスを入力）"
    if not filepath.endswith(".py"):
        return False, "⚠️ .py ファイルのみ書き込み可能です"

    full_path = os.path.join(base_dir, filepath)
    parent_dir = os.path.dirname(full_path)

    if mode == "new" and os.path.exists(full_path):
        return False, f"⚠️ ファイルが既に存在します: {filepath}\n「上書き」または「末尾に追記」を選んでください"

    try:
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(full_path, write_mode, encoding="utf-8") as f:
            if mode == "append":
                f.write(f"\n\n# ── エージェントチーム承認コード ─────────────────────\n")
            f.write(code)
        return True, f"✅ 書き込み完了: `{filepath}`"
    except Exception as e:
        return False, f"❌ 書き込みエラー: {e}"


def _parse_code_changes(text: str) -> list[dict]:
    """鈴木さんの構造化出力から変更リスト [{path, action, reason, code}] を抽出。
    フォーマット:
        ### ファイル: path/to/file.py
        **アクション: 新規作成 | 末尾に追記 | 上書き**
        **理由: ...**
        ```python
        (code)
        ```
    フォーマットが不完全でも可能な限り抽出を試みる。
    """
    changes = []
    # ブロックを "### ファイル:" で分割
    blocks = re.split(r'###\s*ファイル\s*[:：]\s*', text)
    for block in blocks[1:]:  # 先頭は前置きテキストなのでスキップ
        lines = block.strip().splitlines()
        if not lines:
            continue
        path = lines[0].strip().strip('`').strip()
        # .py 拡張子チェック（簡易）
        if not path.endswith('.py'):
            path += '.py'

        # アクション抽出
        action = "末尾に追記"  # デフォルト
        reason = ""
        action_match = re.search(r'\*\*アクション\s*[:：]\s*([^*\n]+)\*\*', block)
        if action_match:
            label = action_match.group(1).strip()
            if "新規" in label:
                action = "新規作成"
            elif "上書" in label:
                action = "上書き"
            else:
                action = "末尾に追記"

        reason_match = re.search(r'\*\*理由\s*[:：]\s*([^*\n]+)\*\*', block)
        if reason_match:
            reason = reason_match.group(1).strip()

        # コードブロック抽出
        code_match = re.search(r'```(?:python)?\n(.*?)```', block, re.DOTALL)
        if not code_match:
            continue
        code = code_match.group(1).strip()

        # セキュリティ: 絶対パス・.. を禁止
        if os.path.isabs(path) or '..' in path.split(os.sep):
            continue

        changes.append({
            'path': path,
            'action': action,
            'reason': reason,
            'code': code,
        })
    return changes


def _render_speech_bubble(name: str, avatar: str, content: str, expandable: bool = False) -> None:
    """キャラクター吹き出しを描画。expandable=True でスマホ向け折りたたみ対応。"""
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(f"**{name}**")
        if expandable and len(content) > 80:
            with st.expander(content[:70] + "…", expanded=False):
                st.markdown(content)
        else:
            st.markdown(content)


def _render_tsune_verdict_card(verdict: dict) -> None:
    """Tuneの決裁カードを描画（承認=緑、修正=黄、却下=赤）。"""
    judgment = verdict.get("judgment", "修正")
    raw = verdict.get("raw", "")
    label = f"{TSUNE['avatar']} **{TSUNE['name']}の決裁: {judgment}**"
    if judgment == "承認":
        st.success(label)
    elif judgment == "修正":
        st.warning(label)
    else:
        st.error(label)
    with st.expander("決裁詳細を読む", expanded=True):
        st.markdown(raw)


def _build_suzuki_code_prompt(round_data: dict) -> str:
    """Tune承認済みのラウンドデータからコード生成プロンプトを構築。"""
    theme = round_data.get("theme", "")
    thread = round_data.get("thread", round_data.get("opinions", []))
    thread_text = _build_thread_context(thread)
    tsune_raw = round_data.get("tsune_verdict", {}).get("raw", "")
    return (
        f"{PERSONA_SUZUKI_CODE}\n\n"
        f"【承認された議論テーマ】\n{theme}\n\n"
        f"【チームの議論スレッド】\n{thread_text}\n\n"
        f"【Tuneの承認内容】\n{tsune_raw}\n\n"
        f"上記の承認内容を実装するPythonコードを生成してください。"
        f"どのファイルのどこに追加するかを明示したうえで、実際に動くコードを書いてください。"
    )


def _render_code_generation(round_data: dict) -> None:
    """承認済みラウンドに対して鈴木さんがコードを生成・自動適用するUI。"""
    round_num = round_data.get("round", 0)
    if not st.session_state.get(SK.AT_CODE_RESULTS):
        st.session_state[SK.AT_CODE_RESULTS] = {}
    if not st.session_state.get(SK.AT_APPLIED_FILES):
        st.session_state[SK.AT_APPLIED_FILES] = {}
    generated = st.session_state[SK.AT_CODE_RESULTS].get(round_num)

    if generated:
        st.markdown(f"#### 💻 鈴木さんの実装コード（ラウンド {round_num}）")

        # 構造化パース（自動適用用）
        changes = _parse_code_changes(generated)

        if changes:
            # ── 自動適用UI ────────────────────────────────────────────
            st.success(f"🔧 **{len(changes)} 個のファイル変更** が検出されました。内容を確認して一括適用できます。")

            base_dir = os.path.dirname(_SCRIPT_DIR)
            applied_history = st.session_state[SK.AT_APPLIED_FILES].get(round_num, [])
            already_applied = {rec["path"] for rec in applied_history}

            # 変更カード一覧
            for idx, ch in enumerate(changes):
                action_emoji = {"新規作成": "🆕", "末尾に追記": "➕", "上書き": "♻️"}.get(ch["action"], "📝")
                status = "✅ 適用済み" if ch["path"] in already_applied else "⏳ 未適用"
                with st.expander(
                    f"{action_emoji} `{ch['path']}` — {ch['action']}  |  {status}",
                    expanded=(ch["path"] not in already_applied),
                ):
                    if ch["reason"]:
                        st.caption(f"💡 {ch['reason']}")
                    st.code(ch["code"], language="python")

            st.divider()
            # 一括適用ボタン
            unapplied = [ch for ch in changes if ch["path"] not in already_applied]
            if unapplied:
                col_apply, col_regen = st.columns([3, 1])
                with col_apply:
                    if st.button(
                        f"🚀 一括適用する（{len(unapplied)} ファイル）",
                        key=f"btn_bulk_apply_{round_num}",
                        type="primary",
                        width='stretch',
                    ):
                        mode_map = {"新規作成": "new", "末尾に追記": "append", "上書き": "overwrite"}
                        results = []
                        for ch in unapplied:
                            ok, msg = _apply_code_to_file(
                                ch["code"], ch["path"],
                                mode_map.get(ch["action"], "append"), base_dir,
                            )
                            results.append((ch["path"], ok, msg))
                            if ok:
                                applied = st.session_state[SK.AT_APPLIED_FILES].setdefault(round_num, [])
                                applied.append({
                                    "path": ch["path"],
                                    "mode": ch["action"],
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                })
                        # 結果サマリ
                        ok_count = sum(1 for _, ok, _ in results if ok)
                        if ok_count == len(results):
                            st.balloons()
                            st.success(f"✅ {ok_count} ファイル全て適用しました！")
                        else:
                            st.warning(f"⚠️ {ok_count}/{len(results)} ファイル適用。エラーあり:")
                            for path, ok, msg in results:
                                if not ok:
                                    st.error(f"`{path}`: {msg}")
                        st.rerun()
                with col_regen:
                    if st.button("🔄 再生成", key=f"btn_regen_{round_num}", width='stretch'):
                        del st.session_state[SK.AT_CODE_RESULTS][round_num]
                        st.rerun()
            else:
                st.success("✅ 全ファイル適用済みです！")
                if st.button("🔄 再生成", key=f"btn_regen_{round_num}"):
                    del st.session_state[SK.AT_CODE_RESULTS][round_num]
                    st.rerun()

        else:
            # フォーマット未検出時 → 従来のコピー表示にフォールバック
            st.caption("⚠️ 自動適用フォーマットが検出されませんでした。コードをコピーして手動で適用してください。")
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", generated, re.DOTALL)
            non_code = re.split(r"```(?:python)?\n.*?```", generated, flags=re.DOTALL)
            for i, text_part in enumerate(non_code):
                if text_part.strip():
                    st.markdown(text_part)
                if i < len(code_blocks):
                    st.code(code_blocks[i].strip(), language="python")
            if not code_blocks:
                st.markdown(generated)
            if st.button("🔄 再生成", key=f"btn_regen_{round_num}"):
                del st.session_state[SK.AT_CODE_RESULTS][round_num]
                st.rerun()

    else:
        if st.button("💻 鈴木さんにコードを生成させる", key=f"btn_code_{round_num}", type="primary"):
            prompt = _build_suzuki_code_prompt(round_data)
            with st.spinner("💻 鈴木さんがコードを書いています... （四苦八苦中）"):
                code_result = _get_llm_response(prompt, timeout_seconds=180)
            if not code_result:
                code_result = "（コードの生成に失敗しました）"
            st.session_state[SK.AT_CODE_RESULTS][round_num] = code_result
            st.rerun()


def _render_round_result(round_data: dict, collapsed: bool = False) -> None:
    """確定済みラウンドを描画。collapsed=True で折りたたみ表示。"""
    round_num = round_data.get("round", "?")
    judgment = round_data.get("tsune_verdict", {}).get("judgment", "")
    judgment_emoji = {"承認": "✅", "修正": "⚠️", "却下": "❌"}.get(judgment, "")
    header = f"ラウンド {round_num} — Tune決裁: {judgment_emoji} {judgment}"
    # "thread" キーを優先、旧 "opinions" にフォールバック
    msgs = round_data.get("thread", round_data.get("opinions", []))

    if collapsed:
        with st.expander(header, expanded=False):
            st.markdown(f"**テーマ:** {round_data.get('theme', '')}")
            st.caption(f"発言数: {len(msgs)}")
            st.divider()
            for msg in msgs:
                _render_speech_bubble(msg["name"], msg["avatar"], msg["content"])
            st.divider()
            _render_tsune_verdict_card(round_data.get("tsune_verdict", {}))
            if judgment in ("承認", "修正"):
                st.divider()
                if judgment == "修正":
                    st.caption("⚠️ Tuneの判定は「修正」ですが、参考コードとして生成できます。")
                _render_code_generation(round_data)
    else:
        st.markdown(f"#### {header}")
        st.markdown(f"**テーマ:** {round_data.get('theme', '')}")


def _render_log_history(logs: list[dict]) -> None:
    """過去の承認済みログ一覧を表示。"""
    if not logs:
        st.info("まだ承認済みの決定事項がありません。")
        return
    st.caption(f"承認済み決定事項: {len(logs)} 件")
    for log in reversed(logs):
        ts = log.get("timestamp", "")[:19].replace("T", " ")
        theme = log.get("theme", "（テーマなし）")[:50]
        round_num = log.get("round", "?")
        with st.expander(f"✅ R{round_num} | {ts} | {theme}…", expanded=False):
            st.markdown(f"**テーマ:** {log.get('theme', '')}")
            msgs = log.get("thread", log.get("opinions", []))
            for msg in msgs:
                with st.chat_message("assistant", avatar=msg.get("avatar", "🤖")):
                    st.markdown(f"**{msg['name']}**")
                    st.markdown(msg.get("content", ""))
            st.divider()
            verdict = log.get("tsune_verdict", {})
            st.success(f"**Tuneの決裁（承認）**\n\n{verdict.get('raw', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ別 UI
# ══════════════════════════════════════════════════════════════════════════════

_PRESET_THEMES: list[dict] = [
    {
        "label": "🔟 改善点10個を検討",
        "theme": (
            "リースシステムの改善すべき点を10個検討してください。\n"
            "各自の専門・立場から「今すぐ直すべき問題」を具体的に挙げ、\n"
            "優先度（高/中/低）と改善方法の概要も添えてください。"
        ),
    },
    {
        "label": "📱 モバイル対応",
        "theme": "審査画面をスマートフォンでも快適に使えるよう改善したい",
    },
    {
        "label": "📊 ダッシュボード改善",
        "theme": "審査スコアのダッシュボードをより直感的でわかりやすくしたい",
    },
    {
        "label": "⚡ 審査スピード改善",
        "theme": "審査判定にかかる時間を短縮し、営業の回転率を上げたい",
    },
]


def _render_phase_a() -> None:
    """フェーズA: テーマ入力と開始ボタン。AT_PENDING が None のときに表示。"""
    # 過去ラウンドを折りたたんで表示
    for past_round in st.session_state.get(SK.AT_HISTORY, []):
        _render_round_result(past_round, collapsed=True)

    st.markdown("---")

    # ── プリセットテーマ ──────────────────────────────────────────────
    st.caption("💡 クイックスタート")
    preset_cols = st.columns(len(_PRESET_THEMES))
    preset_clicked_theme = None
    for i, preset in enumerate(_PRESET_THEMES):
        with preset_cols[i]:
            if st.button(preset["label"], key=f"preset_{i}", width='stretch'):
                preset_clicked_theme = preset["theme"]

    if preset_clicked_theme:
        st.session_state[SK.AT_THEME] = preset_clicked_theme
        st.rerun()

    theme_input = st.text_area(
        "議論テーマ・課題を入力",
        height=100,
        placeholder="例: 審査スコアのダッシュボードをモバイルフレンドリーに改善したい",
        key=SK.AT_THEME,
    )

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        history = st.session_state.get(SK.AT_HISTORY, [])
        label = "🚀 ディスカッション開始" if not history else "🚀 新しいテーマで開始"
        start_btn = st.button(
            label, type="primary",
            disabled=not (theme_input or "").strip(),
            width='stretch',
        )
    with col2:
        clear_btn = st.button("🗑️ 履歴をリセット", width='stretch')
    with col3:
        st.metric("完了ラウンド", len(st.session_state.get(SK.AT_HISTORY, [])))

    if clear_btn:
        st.session_state[SK.AT_HISTORY] = []
        st.session_state[SK.AT_PENDING] = None
        st.rerun()

    if start_btn and (theme_input or "").strip():
        if not is_ai_available():
            st.error("AIが利用できません。サイドバーでエンジンを設定してください。")
            return

        theme = (theme_input or "").strip()
        round_num = len(st.session_state.get(SK.AT_HISTORY, [])) + 1

        st.markdown(f"#### ラウンド {round_num} — 初回ディスカッション開始")
        st.markdown(f"**テーマ:** {theme}")
        st.divider()

        thread: list[dict] = []
        for agent in AGENTS:
            opinion = run_one_agent(agent, theme, thread)
            thread.append(opinion)

        st.session_state[SK.AT_PENDING] = {
            "theme": theme,
            "round_num": round_num,
            "thread": thread,
        }
        st.rerun()


def _render_phase_b(pending: dict) -> None:
    """フェーズB: ディスカッション中のUI。ボタンを上部に固定してスレッドを下に表示。"""
    theme = pending["theme"]
    thread: list[dict] = pending["thread"]
    round_num = pending["round_num"]

    st.markdown(f"**テーマ:** {theme}")
    st.caption(f"ラウンド {round_num} — 発言数: {len(thread)} 件")

    # リバタルモード表示
    rebuttal_target = st.session_state.get(SK.AT_REBUTTAL_TARGET)
    if rebuttal_target:
        st.info(
            f"💬 **リバタルモード** — 「{rebuttal_target['name']}」の発言に反応させます。"
            f"「誰に発言させますか？」からエージェントを選んでください。"
        )
        if st.button("❌ リバタルモード解除", key="btn_cancel_rebuttal"):
            st.session_state[SK.AT_REBUTTAL_TARGET] = None
            st.rerun()

    # ── ボタンゾーン（常に上部に表示）────────────────────────────────────────
    st.markdown("#### 誰に発言させますか？")

    # 4エージェントを 2×2 グリッドで表示
    cols = st.columns(2)
    clicked_agent = None
    for i, agent in enumerate(AGENTS):
        with cols[i % 2]:
            if st.button(
                f"{agent['avatar']} {agent['name']}に発言させる",
                key=f"btn_speak_{agent['id']}",
                width='stretch',
            ):
                clicked_agent = agent

    # 自動ラウンド進行
    st.markdown("#### 自動ラウンド進行")
    col_auto1, col_auto2 = st.columns([2, 1])
    with col_auto1:
        auto_rounds = st.slider("ラウンド数", min_value=1, max_value=3, value=2, key="auto_round_slider")
    with col_auto2:
        auto_round_btn = st.button(
            f"🔄 自動議論（{auto_rounds}回）",
            key="btn_auto_rounds",
            width='stretch',
        )

    # Tune（意見・論点整理・決裁）ボタン
    col_op, col_fac, col_verdict, col_reset = st.columns([2, 2, 3, 1])
    with col_op:
        tsune_opinion_btn = st.button(
            "💬 Tuneに意見を言わせる",
            key="btn_tsune_opinion",
            width='stretch',
        )
    with col_fac:
        tsune_facilitate_btn = st.button(
            "🔍 論点整理（Tune）",
            key="btn_tsune_facilitate",
            width='stretch',
        )
    with col_verdict:
        tsune_verdict_btn = st.button(
            "👔 Tuneに決裁させる",
            key="btn_tsune_verdict",
            type="primary",
            width='stretch',
        )
    with col_reset:
        reset_btn = st.button("🗑️ リセット", key="btn_reset_b", width='stretch')

    st.divider()

    # ── スレッド表示（ラウンド区切り線付き）──────────────────────────────────
    displayed_rounds: set[int] = set()
    for msg in thread:
        msg_round = msg.get("round")
        if msg_round is not None and msg_round not in displayed_rounds:
            displayed_rounds.add(msg_round)
            st.markdown(f"**━━━ ラウンド {msg_round} ━━━**")
        _render_speech_bubble(msg["name"], msg["avatar"], msg["content"])
        # 💬 反論する ボタン
        if st.button(
            f"💬 反論する",
            key=f"btn_rebuttal_{msg.get('turn', id(msg))}",
            help=f"「{msg['name']}」の発言にリバタルする",
        ):
            st.session_state[SK.AT_REBUTTAL_TARGET] = msg
            st.rerun()

    # 論点整理結果を表示
    facilitator_result = st.session_state.get(SK.AT_FACILITATOR_RESULT)
    if facilitator_result:
        st.divider()
        st.markdown(f"#### 🔍 {TSUNE['avatar']} Tune の論点整理")
        st.info(facilitator_result)

    # ── ボタン処理（描画後に実行）─────────────────────────────────────────────
    if clicked_agent is not None:
        cur_rebuttal_target = st.session_state.get(SK.AT_REBUTTAL_TARGET)
        is_rebuttal = cur_rebuttal_target is not None
        opinion = run_one_agent(
            clicked_agent, theme, thread,
            target_speaker=cur_rebuttal_target,
            rebuttal_mode=is_rebuttal,
        )
        pending["thread"].append(opinion)
        st.session_state[SK.AT_PENDING] = pending
        st.session_state[SK.AT_REBUTTAL_TARGET] = None
        st.rerun()

    if auto_round_btn:
        current_round = max((m.get("round", 0) for m in thread), default=0)
        for r in range(1, auto_rounds + 1):
            run_debate_round(theme, thread, AGENTS, current_round + r)
        pending["thread"] = thread
        st.session_state[SK.AT_PENDING] = pending
        st.rerun()

    if tsune_opinion_btn:
        tsune_op = run_tsune_opinion(theme, thread)
        pending["thread"].append(tsune_op)
        st.session_state[SK.AT_PENDING] = pending
        st.rerun()

    if tsune_facilitate_btn:
        result_text = run_tune_facilitate(theme, thread)
        st.session_state[SK.AT_FACILITATOR_RESULT] = result_text
        st.rerun()

    if tsune_verdict_btn:
        # 論点整理を自動挿入
        st.divider()
        st.markdown(f"#### 🔍 {TSUNE['avatar']} Tune 論点整理（決裁前）")
        fac_text = run_tune_facilitate(theme, thread)
        st.info(fac_text)
        st.session_state[SK.AT_FACILITATOR_RESULT] = fac_text
        st.divider()

        result = run_tsune_verdict(theme, round_num, thread)
        st.session_state[SK.AT_HISTORY] = st.session_state.get(SK.AT_HISTORY, []) + [result]
        st.session_state[SK.AT_PENDING] = None
        st.session_state[SK.AT_FACILITATOR_RESULT] = None
        if result["tsune_verdict"]["approved"]:
            save_agent_team_log(result)
            st.balloons()
            st.success("🎉 Tuneが承認しました！決定事項ログに保存されました。")
            st.info("👇 履歴を開いて「鈴木さんにコードを生成させる」ボタンを押してください。")
        else:
            judgment = result["tsune_verdict"]["judgment"]
            if judgment == "修正":
                st.warning("⚠️ 修正して継続してください。")
            else:
                st.error("❌ 却下されました。テーマや方針を見直してください。")

        # ── Slack 自動通知 ──────────────────────────────────────────
        if st.session_state.get(SK.SLACK_NOTIFY_VERDICT, True) and _is_slack_enabled():
            slack_ok, slack_msg = send_slack_notification(result)
            if slack_ok:
                st.toast("📡 Slackに通知しました", icon="✅")
            else:
                st.toast(f"Slack通知失敗: {slack_msg}", icon="⚠️")

        st.rerun()

    if reset_btn:
        st.session_state[SK.AT_PENDING] = None
        st.session_state[SK.AT_HISTORY] = []
        st.session_state[SK.AT_REBUTTAL_TARGET] = None
        st.session_state[SK.AT_FACILITATOR_RESULT] = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# メインエントリポイント
# ══════════════════════════════════════════════════════════════════════════════

def _render_discussion_tab() -> None:
    """議論タブの内部描画。フェーズA/Bを切り替える。"""
    # セッション初期化
    if SK.AT_HISTORY not in st.session_state:
        st.session_state[SK.AT_HISTORY] = []
    if SK.AT_PENDING not in st.session_state:
        st.session_state[SK.AT_PENDING] = None
    if SK.AT_CODE_RESULTS not in st.session_state:
        st.session_state[SK.AT_CODE_RESULTS] = {}
    if SK.AT_APPLIED_FILES not in st.session_state:
        st.session_state[SK.AT_APPLIED_FILES] = {}
    if SK.AT_REBUTTAL_TARGET not in st.session_state:
        st.session_state[SK.AT_REBUTTAL_TARGET] = None
    if SK.AT_FACILITATOR_RESULT not in st.session_state:
        st.session_state[SK.AT_FACILITATOR_RESULT] = None

    pending = st.session_state.get(SK.AT_PENDING)

    if pending is None:
        # フェーズA: テーマ入力
        _render_phase_a()
    else:
        # フェーズB: ディスカッション中
        _render_phase_b(pending)


def _render_slack_settings() -> None:
    """Slack連携設定タブのUI。"""
    st.markdown("### 📡 Slack 連携設定")
    st.caption(
        "Slack Incoming Webhook を使って、Tuneの決裁結果をSlackチャンネルに自動通知します。"
    )

    # ── Webhook URL 設定 ──────────────────────────────────────────────
    st.markdown("#### 1. Webhook URL")
    st.caption(
        "Slack App → Incoming Webhooks → Webhook URLをコピーして貼り付けてください。\n"
        "[Slack Webhook 設定ガイド](https://api.slack.com/messaging/webhooks)"
    )

    current_url = st.session_state.get(SK.SLACK_WEBHOOK_URL, "")
    webhook_input = st.text_input(
        "Webhook URL",
        value=current_url,
        type="password",
        placeholder="Slack Incoming Webhook URL を貼り付け",
        key="slack_webhook_input",
    )

    # URL が変わったら即保存
    if webhook_input != current_url:
        st.session_state[SK.SLACK_WEBHOOK_URL] = webhook_input

    # URL バリデーション
    if webhook_input:
        if webhook_input.startswith("https://hooks.slack.com/"):
            st.success("✅ Webhook URL の形式は正しいです")
        else:
            st.error("⚠️ URLは `https://hooks.slack.com/` で始まる必要があります")

    st.markdown("---")

    # ── 通知設定 ──────────────────────────────────────────────────────
    st.markdown("#### 2. 通知設定")

    col1, col2 = st.columns(2)
    with col1:
        slack_enabled = st.toggle(
            "Slack 通知を有効にする",
            value=st.session_state.get(SK.SLACK_ENABLED, False),
            key="slack_enabled_toggle",
        )
        st.session_state[SK.SLACK_ENABLED] = slack_enabled

    with col2:
        notify_thread = st.toggle(
            "議論スレッドも含める",
            value=st.session_state.get(SK.SLACK_NOTIFY_THREAD, True),
            key="slack_notify_thread_toggle",
            help="ONにすると全メンバーの発言がSlack通知に含まれます",
        )
        st.session_state[SK.SLACK_NOTIFY_THREAD] = notify_thread

    notify_verdict = st.toggle(
        "Tuneの決裁時に自動通知",
        value=st.session_state.get(SK.SLACK_NOTIFY_VERDICT, True),
        key="slack_notify_verdict_toggle",
        help="ONにすると、Tuneが決裁を下した時点で自動的にSlackに通知されます",
    )
    st.session_state[SK.SLACK_NOTIFY_VERDICT] = notify_verdict

    channel_name = st.text_input(
        "通知先チャンネル名（メモ用）",
        value=st.session_state.get(SK.SLACK_CHANNEL_NAME, ""),
        placeholder="#lease-review-team",
        key="slack_channel_name_input",
        help="Webhook に紐付いたチャンネル名をメモとして記録します（通知先はWebhook側で決まります）",
    )
    st.session_state[SK.SLACK_CHANNEL_NAME] = channel_name

    st.markdown("---")

    # ── テスト送信 ──────────────────────────────────────────────────
    st.markdown("#### 3. テスト送信")

    test_col1, test_col2 = st.columns([3, 1])
    with test_col1:
        test_message = st.text_input(
            "テストメッセージ",
            value="🧪 リース審査システム — Slack連携テスト送信です！",
            key="slack_test_msg",
        )
    with test_col2:
        st.markdown("")  # 高さ調整
        st.markdown("")
        test_btn = st.button(
            "📨 テスト送信",
            key="btn_slack_test",
            type="primary",
            width='stretch',
            disabled=not _is_slack_enabled(),
        )

    if test_btn:
        with st.spinner("Slackに送信中..."):
            ok, msg = send_slack_custom_message(test_message)
        if ok:
            st.success(msg)
            st.balloons()
        else:
            st.error(msg)

    if not _is_slack_enabled():
        if not webhook_input:
            st.info("💡 Webhook URL を入力してください。")
        elif not slack_enabled:
            st.info("💡 「Slack通知を有効にする」をONにしてください。")
        else:
            st.warning("⚠️ Webhook URL の形式が正しくありません。")

    st.markdown("---")

    # ── 手動送信（過去ラウンドの通知）──────────────────────────────
    st.markdown("#### 4. 過去の決裁結果を手動送信")
    st.caption("過去のラウンド結果を選んでSlackに手動送信できます。")

    history = st.session_state.get(SK.AT_HISTORY, [])
    if history:
        options = [
            f"ラウンド {r.get('round', '?')} — {r.get('tsune_verdict', {}).get('judgment', '?')} — {r.get('theme', '')[:40]}"
            for r in history
        ]
        selected_idx = st.selectbox(
            "送信するラウンド",
            range(len(options)),
            format_func=lambda i: options[i],
            key="slack_manual_select",
        )
        if st.button(
            "📨 このラウンドの結果をSlackに送信",
            key="btn_slack_manual",
            disabled=not _is_slack_enabled(),
        ):
            with st.spinner("Slackに送信中..."):
                ok, msg = send_slack_notification(history[selected_idx])
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.info("まだ決裁済みのラウンドがありません。")


# ══════════════════════════════════════════════════════════════════════════════
# 審査討論強化ヘルパー群
# ══════════════════════════════════════════════════════════════════════════════

# ── Tune ファシリテーター v2（賛否割合・最重要リスク3点・推奨アクション）────────
PERSONA_SHINSA_FACILITATOR_V2 = """あなたは「統括マネージャーTune✨」のファシリテーターモードです。
討論スレッドを読み、以下の形式で必ず出力してください。

## 討論サマリー

**賛否の割合:**
承認寄り: X人 / 条件付き承認: Y人 / 否決寄り: Z人
（スレッド内の「【承認】」「【条件付き承認】」「【否決】」の出現を集計する）

**最重要リスク3点:**
1. 〈第1リスク — 具体的な数値や根拠を含めること〉
2. 〈第2リスク〉
3. 〈第3リスク〉

**推奨アクション:**
- 〈具体的なアクション1〉
- 〈具体的なアクション2〉
- 〈具体的なアクション3（任意）〉

**Tuneの総括:**
〈Tuneとしての状況判断。60字以内〉"""

# ── Tune 最終評決プロンプト ────────────────────────────────────────────────
PERSONA_SHINSA_FINAL_VERDICT = """あなたは「統括マネージャーTune✨」です。
討論全体を踏まえて最終評決を下します。

【必須出力形式 — 必ずこの形式で出力すること】

## 最終評決

**評決: 承認** または **評決: 条件付き承認** または **評決: 否決**
（必ずこの3択のいずれかを選ぶこと）

**評決理由:**
〈2〜3文で簡潔に〉

**スコア調整の根拠:**
〈討論結果がスコアに与える影響を説明〉
調整値: +X点 または -X点（-10〜+10の範囲で具体的な数字を出すこと）

（末尾は必ず「以上。このプロジェクトは極秘だ。」で締めること）"""


def _make_stream_generator(text: str):
    """テキストをword単位のストリーム表示ジェネレーターに変換。"""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.022)


def _build_round_summary(opinions: list[dict], round_num: int) -> str:
    """1ラウンド分の発言リストをサマリーテキストに変換（次ラウンドへ引き継ぎ用）。"""
    if not opinions:
        return ""
    lines = [f"=== ラウンド{round_num} サマリー ==="]
    for op in opinions:
        verdict_tag = ""
        content = op.get("content", "")
        for tag in ["【否決】", "【条件付き承認】", "【承認】"]:
            if tag in content[:30]:
                verdict_tag = tag
                break
        excerpt = content[:80].replace("\n", " ")
        lines.append(f"・{op['name']}({op['avatar']}) {verdict_tag} → {excerpt}…")
    return "\n".join(lines)


def _build_shinsa_agent_prompt_v2(
    agent: dict,
    theme: str,
    thread: list[dict],
    prev_round_summary: str = "",
    target_speaker: dict | None = None,
    round_num: int = 1,
) -> str:
    """審査討論専用プロンプト（引用反論・数値根拠・前ラウンドサマリー引き継ぎ）。"""
    thread_text = _build_thread_context(thread)
    specialty = agent.get("specialty", "担当分野")

    summary_block = (
        f"\n\n【前ラウンドのサマリー（必ず参照すること）】\n{prev_round_summary}\n"
        if prev_round_summary else ""
    )

    if round_num == 1:
        instruction = (
            f"{agent['full_name']}として、上記審査案件について初回意見を述べてください。\n"
            f"専門領域「{specialty}」の観点から具体的な根拠を示し、\n"
            f"冒頭に「【承認】」「【条件付き承認】」「【否決】」のいずれかを必ず明示してください。"
        )
    else:
        if target_speaker:
            t_name = target_speaker["name"]
            t_content = target_speaker["content"][:120]
            instruction = (
                f"{agent['full_name']}として、{t_name}の発言に対して引用形式で反論・補足してください。\n\n"
                f"必須ルール:\n"
                f"1. 「{t_name}は『{t_content}…』と述べたが、{specialty}の観点では…」という形式で始めること\n"
                f"2. 具体的な数値（比率・業界平均・パーセンテージ等）を必ず含めること\n"
                f"3. 冒頭に「【承認】」「【条件付き承認】」「【否決】」のいずれかを明示すること"
            )
        else:
            instruction = (
                f"{agent['full_name']}として、前ラウンドの議論を踏まえて追加意見を述べてください。\n"
                f"専門領域「{specialty}」の数値根拠を必ず含め、冒頭に承認/条件付き承認/否決を明示すること。"
            )

    return (
        f"{agent['persona']}\n\n"
        f"【審査案件】\n{theme}\n"
        f"{summary_block}"
        f"\n【これまでの議論スレッド】\n{thread_text if thread_text else '（まだ発言なし）'}\n"
        f"\n{instruction}"
    )


def run_shinsa_one_agent_v2(
    agent: dict,
    theme: str,
    thread: list[dict],
    prev_round_summary: str = "",
    target_speaker: dict | None = None,
    round_num: int = 1,
    streaming: bool = True,
) -> dict:
    """審査討論用：1エージェント呼び出しとストリーミング表示。"""
    prompt = _build_shinsa_agent_prompt_v2(
        agent, theme, thread, prev_round_summary, target_speaker, round_num
    )
    with st.spinner(f"{agent['avatar']} {agent['name']}が考えています..."):
        content = _get_llm_response(prompt, timeout_seconds=120)
    if not content:
        content = "（応答がありませんでした）"

    opinion: dict = {
        "agent_id": agent["id"],
        "name": agent["name"],
        "avatar": agent["avatar"],
        "content": content,
        "turn": len(thread) + 1,
        "round": round_num,
    }

    # 吹き出し表示
    with st.chat_message("assistant", avatar=agent["avatar"]):
        st.markdown(f"**{agent['name']}** _{agent.get('specialty', '')}専門_")
        if streaming:
            st.write_stream(_make_stream_generator(content))
        else:
            with st.expander("発言を読む", expanded=True):
                st.markdown(content)

    return opinion


def run_shinsa_debate_round_v2(
    theme: str,
    thread: list[dict],
    agents: list[dict],
    round_num: int,
    prev_round_summary: str = "",
    streaming: bool = True,
) -> tuple[list[dict], str]:
    """審査討論1ラウンドを実行し (new_opinions, round_summary) を返す。"""
    import random as _random
    prev_round_msgs = [m for m in thread if m.get("round") == round_num - 1]
    new_opinions: list[dict] = []

    st.markdown(f"**━━━━ ラウンド {round_num} ━━━━**")

    for agent in agents:
        target: dict | None = None
        if round_num >= 2 and prev_round_msgs:
            candidates = [m for m in prev_round_msgs if m.get("agent_id") != agent["id"]]
            if candidates:
                target = _random.choice(candidates)

        opinion = run_shinsa_one_agent_v2(
            agent, theme, thread, prev_round_summary, target, round_num, streaming
        )
        thread.append(opinion)
        new_opinions.append(opinion)

    summary = _build_round_summary(new_opinions, round_num)
    return new_opinions, summary


def run_shinsa_facilitator_v2(theme: str, thread: list[dict]) -> str:
    """Tuneが賛否割合・最重要リスク3点・推奨アクションを含むサマリーを出力。"""
    thread_text = _build_thread_context(thread)
    prompt = (
        f"{PERSONA_SHINSA_FACILITATOR_V2}\n\n"
        f"【審査案件テーマ】\n{theme}\n\n"
        f"【議論スレッド】\n{thread_text}\n\n"
        f"上記の討論を整理してください。必ず指定のフォーマットで出力してください。"
    )
    with st.spinner(f"✨ Tuneが討論サマリーを作成しています..."):
        result = _get_llm_response(prompt, timeout_seconds=120)
    if not result:
        result = "## 討論サマリー\n（整理できませんでした）"
    return result


def run_shinsa_final_verdict_auto(theme: str, thread: list[dict], facilitator_text: str) -> dict:
    """Tuneが最終評決（承認/条件付き承認/否決）を自動出力。"""
    thread_text = _build_thread_context(thread)
    prompt = (
        f"{PERSONA_SHINSA_FINAL_VERDICT}\n\n"
        f"【審査案件テーマ】\n{theme}\n\n"
        f"【討論スレッド】\n{thread_text}\n\n"
        f"【討論サマリー】\n{facilitator_text}\n\n"
        f"上記を踏まえ、最終評決を出してください。"
    )
    with st.spinner("✨ Tuneが最終評決を下しています..."):
        raw = _get_llm_response(prompt, timeout_seconds=120)
    if not raw:
        raw = "## 最終評決\n**評決: 条件付き承認**\n（応答がありませんでした）\n以上。このプロジェクトは極秘だ。"

    # 評決種別を抽出
    final_label = "条件付き承認"
    for label in ["否決", "条件付き承認", "承認"]:
        if f"評決: {label}" in raw or f"評決:{label}" in raw:
            final_label = label
            break

    return {"raw": raw, "final_label": final_label}


def _calc_debate_score_adjustment(thread: list[dict], final_verdict: dict) -> int:
    """討論結果からリスクスコア調整値を算出（-10〜+10点）。

    承認寄りなら上方修正、否決寄りなら下方修正。
    """
    approve = 0
    conditional = 0
    reject = 0

    for msg in thread:
        content = msg.get("content", "")
        head = content[:40]
        if "【否決】" in head:
            reject += 1
        elif "【条件付き承認】" in head:
            conditional += 1
        elif "【承認】" in head:
            approve += 1

    total = approve + conditional + reject
    if total == 0:
        base = 0
    else:
        weighted = approve * 1.0 + conditional * 0.4 + reject * (-1.0)
        ratio = weighted / total
        base = round(ratio * 8)  # -8〜+8

    # 最終評決で補正
    label = final_verdict.get("final_label", "条件付き承認")
    if label == "否決":
        base = max(base - 3, -10)
    elif label == "承認":
        base = min(base + 2, 10)

    return max(-10, min(10, base))


def _render_shinsa_history_item(past: dict) -> None:
    """審査討論の過去履歴を折りたたみ表示。"""
    final_v = past.get("final_verdict", {})
    final_label = final_v.get("final_label", past.get("tsune_verdict", {}).get("judgment", "?"))
    label_emoji = {"承認": "✅", "条件付き承認": "⚠️", "否決": "❌"}.get(final_label, "❓")
    with st.expander(f"{label_emoji} 審査討論 — 最終評決: {final_label}", expanded=False):
        for msg in past.get("thread", []):
            msg_round = msg.get("round")
            with st.chat_message("assistant", avatar=msg.get("avatar", "🤖")):
                st.markdown(f"**{msg['name']}**")
                with st.expander(msg["content"][:60] + "...", expanded=False):
                    st.markdown(msg["content"])
        st.divider()
        fac = past.get("facilitator_text", "")
        if fac:
            st.info(fac)
        raw_verdict = final_v.get("raw", past.get("tsune_verdict", {}).get("raw", ""))
        if raw_verdict:
            st.success(f"✨ **Tuneの最終評決**\n\n{raw_verdict}")


def _render_shinsa_debate_tab() -> None:
    """⑤ 審査討論モードタブ（強化版） — 軍師🎖️・Dr.Algo🔬・タム🏦がN回自動討論し、Tune✨が集約。"""
    st.markdown("### 🏦 審査討論モード")
    st.caption(
        "軍師🎖️（戦略リスク）・Dr.Algo🔬（統計・数値）・タム🏦（現場経験）が数値根拠付きで討論。\n"
        "Tune✨がサマリー＋最終評決を出力し、審査スコアを自動調整します。"
    )

    # セッション初期化
    if SK.AT_SHINSA_HISTORY not in st.session_state:
        st.session_state[SK.AT_SHINSA_HISTORY] = []
    if SK.AT_SHINSA_PENDING not in st.session_state:
        st.session_state[SK.AT_SHINSA_PENDING] = None

    res = st.session_state.get(SK.LAST_RESULT)
    if not res:
        st.info("審査データがありません。先に審査を実行してください。")
        return

    # ── 案件概要 & スコア調整値 ─────────────────────────────────────────
    score    = res.get("score", 0)
    hantei   = res.get("hantei", "—")
    industry = res.get("industry_sub", "—")
    asset    = res.get("asset_name", "—")

    col_sc, col_adj, col_info = st.columns([2, 2, 3])
    with col_sc:
        st.metric("審査スコア", f"{score:.0f}点", delta=hantei)
    with col_adj:
        adj = st.session_state.get("debate_score_adjustment")
        if adj is not None:
            delta_str = f"+{adj}" if adj > 0 else str(adj)
            color = "normal" if adj == 0 else ("normal" if adj > 0 else "inverse")
            st.metric("討論スコア調整値", f"{delta_str}点", delta=adj, delta_color=color)
        else:
            st.metric("討論スコア調整値", "未実施", delta=None)
    with col_info:
        st.caption(f"業種: {industry}")
        st.caption(f"物件: {asset}")

    st.divider()

    shinsa_pending = st.session_state.get(SK.AT_SHINSA_PENDING)
    shinsa_history = st.session_state.get(SK.AT_SHINSA_HISTORY, [])

    # ── 過去の審査討論履歴 ──────────────────────────────────────────────
    for past in shinsa_history:
        _render_shinsa_history_item(past)

    # ── 討論未開始 ──────────────────────────────────────────────────────
    if shinsa_pending is None:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            rounds_n = st.slider("討論ラウンド数", min_value=1, max_value=3, value=2, key="shinsa_rounds_n")
        with col2:
            st.markdown("")
            start_btn = st.button(
                f"🏦 審査討論開始（自動{rounds_n}ラウンド）",
                type="primary", key="btn_shinsa_start", width="stretch",
            )
        with col3:
            st.markdown("")
            if st.button("🗑️ 履歴リセット", key="btn_shinsa_reset", width="stretch"):
                st.session_state[SK.AT_SHINSA_HISTORY] = []
                st.session_state["debate_score_adjustment"] = None
                st.rerun()

        if start_btn:
            from ai_chat import is_ai_available
            if not is_ai_available():
                st.error("AIが利用できません。サイドバーでエンジンを設定してください。")
                return

            theme = _build_shinsa_theme(res)
            thread: list[dict] = []
            prev_summary = ""

            for r in range(1, rounds_n + 1):
                _, prev_summary = run_shinsa_debate_round_v2(
                    theme, thread, SHINSA_DEBATE_AGENTS,
                    round_num=r, prev_round_summary=prev_summary, streaming=True,
                )

            st.session_state[SK.AT_SHINSA_PENDING] = {
                "theme": theme,
                "thread": thread,
                "round_num": rounds_n,
            }
            st.rerun()

    # ── 討論完了・Tune集約待ち ───────────────────────────────────────────
    else:
        theme     = shinsa_pending["theme"]
        thread    = shinsa_pending["thread"]
        round_num = shinsa_pending["round_num"]

        st.caption(f"{round_num}ラウンド完了 — 発言数: {len(thread)} 件")

        # スレッド再表示（expander折りたたみ）
        displayed_rounds: set[int] = set()
        for msg in thread:
            msg_round = msg.get("round")
            if msg_round is not None and msg_round not in displayed_rounds:
                displayed_rounds.add(msg_round)
                st.markdown(f"**━━━ ラウンド {msg_round} ━━━**")
            _render_speech_bubble(msg["name"], msg["avatar"], msg["content"], expandable=True)

        st.divider()

        col_fac, col_verdict, col_reset = st.columns([3, 3, 1])
        with col_fac:
            if st.button("🔍 Tuneがサマリー作成", key="btn_shinsa_fac", width="stretch"):
                fac_text = run_shinsa_facilitator_v2(theme, thread)
                st.session_state["_shinsa_fac_text"] = fac_text
                st.rerun()
        with col_verdict:
            if st.button("✨ 最終評決 & スコア調整", type="primary", key="btn_shinsa_final_verdict", width="stretch"):
                # ファシリテーターサマリーがなければ先に生成
                fac_text = st.session_state.get("_shinsa_fac_text") or run_shinsa_facilitator_v2(theme, thread)
                st.session_state["_shinsa_fac_text"] = fac_text

                # 最終評決
                final_v = run_shinsa_final_verdict_auto(theme, thread, fac_text)

                # スコア調整値算出 & 保存
                adj_val = _calc_debate_score_adjustment(thread, final_v)
                st.session_state["debate_score_adjustment"] = adj_val

                # 履歴保存
                record = {
                    "theme": theme,
                    "thread": thread,
                    "round_num": round_num,
                    "facilitator_text": fac_text,
                    "final_verdict": final_v,
                    "score_adjustment": adj_val,
                    "timestamp": datetime.datetime.now().isoformat(),
                    # 旧フォーマット互換
                    "tsune_verdict": {
                        "raw": final_v.get("raw", ""),
                        "judgment": final_v.get("final_label", "条件付き承認"),
                        "approved": final_v.get("final_label") == "承認",
                    },
                }
                st.session_state[SK.AT_SHINSA_HISTORY] = shinsa_history + [record]
                st.session_state[SK.AT_SHINSA_PENDING] = None
                st.session_state["_shinsa_fac_text"] = None
                st.rerun()
        with col_reset:
            if st.button("🔄 再討論", key="btn_shinsa_rerun", width="stretch"):
                st.session_state[SK.AT_SHINSA_PENDING] = None
                st.session_state["_shinsa_fac_text"] = None
                st.rerun()

        # ファシリテーターサマリー表示
        fac_text = st.session_state.get("_shinsa_fac_text")
        if fac_text:
            st.divider()
            st.markdown("#### ✨ Tune の討論サマリー")
            st.info(fac_text)


def render_agent_team() -> None:
    """エージェントチーム議論ページのメインエントリポイント。"""
    st.title("🤝 エージェントチーム議論")
    st.caption(
        "4人が自由にディスカッション → 準備ができたら **Tune** に決裁を依頼。"
        "Tuneの承認なしには決定事項に移行できません。"
    )

    engine = st.session_state.get(SK.AI_ENGINE, "ollama")
    slack_status = "🟢 Slack通知ON" if _is_slack_enabled() else "⚪ Slack通知OFF"
    st.caption(f"🤖 使用中: {'Gemini API' if engine == 'gemini' else 'Ollama（ローカル）'}　|　📡 {slack_status}")

    tab_discuss, tab_shinsa, tab_logs, tab_slack = st.tabs(
        ["💬 議論", "🏦 審査討論モード", "📋 決定事項ログ", "📡 Slack連携"]
    )

    with tab_discuss:
        _render_discussion_tab()

    with tab_shinsa:
        _render_shinsa_debate_tab()

    with tab_logs:
        logs = load_agent_team_logs()
        _render_log_history(logs)

    with tab_slack:
        _render_slack_settings()
