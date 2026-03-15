# -*- coding: utf-8 -*-
"""
components/agent_team.py
========================
エージェントチーム議論モジュール。

5人のキャラクター付きエージェントが自由にディスカッションし、
統括マネージャー「つね」が最終決裁を行う。
つねの承認なしには決定事項に移行できない。

使い方:
    from components.agent_team import render_agent_team
    render_agent_team()
"""

from __future__ import annotations

import json
import re
import datetime
import os
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
つねに承認された開発仕様を受け取り、実際に動くPythonコードを生成します。

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

PERSONA_TSUNE_OPINION = """あなたは「統括マネージャーつね（つね）」です。
今回は決裁者としてではなく、一管理職として自分の意見を述べます。
このプロジェクトは極秘なので、発言には細心の注意を払ってください。

【キャラクター】
・「管理者の立場から言うと…」で始めるクセがある
・現場の状況・リスク・優先順位の観点から意見を言う
・たまに「決裁はまだだぞ」と念を押す

【発言ルール】
・参考意見として述べること（「承認」「却下」などの判定語は使わない）
・200文字以内。「まあ、最終的には私が決める話ですけどね」で締めること"""

PERSONA_TSUNE = """あなたは「統括マネージャーつね（つね）」です。
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
    "id": "tsune", "name": "つね", "full_name": "統括マネージャーつね", "avatar": "👔", "persona": PERSONA_TSUNE,
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
        lines.append(f"--- ラウンド{rd['round']}（つね判定: {judgment}）---")
        msgs = rd.get("thread", rd.get("opinions", []))
        for msg in msgs[:2]:  # コンテキスト節約のため先頭2件のみ
            lines.append(f"  {msg['name']}: {msg['content'][:80]}…")
    return "\n".join(lines)


def _build_agent_prompt(agent: dict, theme: str, thread: list[dict]) -> str:
    """各エージェント用プロンプトを構築。スレッド全体を読んで返答させる。"""
    history_text = _build_history_context()
    thread_text = _build_thread_context(thread)
    history_block = f"\n\n【過去の議論（参考）】\n{history_text}\n" if history_text else ""
    thread_block = f"\n\n【現在の議論スレッド】\n{thread_text}\n" if thread_text else ""
    instruction = (
        f"上記の議論スレッドを踏まえて、{agent['full_name']}として追加コメントを述べてください。"
        if thread else
        f"{agent['full_name']}として、上記テーマについて初回の意見を述べてください。"
    )
    return (
        f"{agent['persona']}\n\n"
        f"【議論テーマ】\n{theme}\n"
        f"{history_block}"
        f"{thread_block}"
        f"\n{instruction}"
    )


def _build_tsune_prompt(theme: str, thread: list[dict]) -> str:
    """つね決裁用プロンプト（スレッド全体を集約して判定）。"""
    thread_text = _build_thread_context(thread)
    return (
        f"{PERSONA_TSUNE}\n\n"
        f"【議論テーマ】\n{theme}\n\n"
        f"【議論スレッド全体】\n{thread_text}\n\n"
        f"上記の議論を受けて、承認・修正・却下の判定を出してください。"
        f"必ず「判定: 承認」「判定: 修正」「判定: 却下」のいずれかで始めてください。"
    )


def _parse_tsune_judgment(text: str) -> str:
    """つねの応答テキストから判定文字列を抽出。"""
    for keyword in ["承認", "却下", "修正"]:
        if f"判定: {keyword}" in text or f"判定:{keyword}" in text:
            return keyword
    for keyword in ["承認", "却下", "修正"]:
        if keyword in text[:100]:
            return keyword
    return "修正"


def run_one_agent(agent: dict, theme: str, thread: list[dict]) -> dict:
    """1エージェントを呼び出して発言を返す。吹き出しもここで描画する。"""
    prompt = _build_agent_prompt(agent, theme, thread)
    with st.spinner(f"{agent['avatar']} {agent['name']}が考えています..."):
        content = _get_llm_response(prompt, timeout_seconds=120)
    if not content:
        content = "（応答がありませんでした）"
    turn = len(thread) + 1
    opinion = {
        "agent_id": agent["id"],
        "name": agent["name"],
        "avatar": agent["avatar"],
        "content": content,
        "turn": turn,
    }
    _render_speech_bubble(agent["name"], agent["avatar"], content)
    return opinion


def run_tsune_opinion(theme: str, thread: list[dict]) -> dict:
    """つねが参加者として意見を述べる（決裁ではない）。"""
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
    """つねが最終決裁を行い、ラウンド結果 dict を返す。"""
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


def _render_speech_bubble(name: str, avatar: str, content: str) -> None:
    """キャラクター吹き出しを描画。"""
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(f"**{name}**")
        st.markdown(content)


def _render_tsune_verdict_card(verdict: dict) -> None:
    """つねの決裁カードを描画（承認=緑、修正=黄、却下=赤）。"""
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
    """つね承認済みのラウンドデータからコード生成プロンプトを構築。"""
    theme = round_data.get("theme", "")
    thread = round_data.get("thread", round_data.get("opinions", []))
    thread_text = _build_thread_context(thread)
    tsune_raw = round_data.get("tsune_verdict", {}).get("raw", "")
    return (
        f"{PERSONA_SUZUKI_CODE}\n\n"
        f"【承認された議論テーマ】\n{theme}\n\n"
        f"【チームの議論スレッド】\n{thread_text}\n\n"
        f"【つねの承認内容】\n{tsune_raw}\n\n"
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
                        use_container_width=True,
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
                    if st.button("🔄 再生成", key=f"btn_regen_{round_num}", use_container_width=True):
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
    header = f"ラウンド {round_num} — つね決裁: {judgment_emoji} {judgment}"
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
                    st.caption("⚠️ つねの判定は「修正」ですが、参考コードとして生成できます。")
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
            st.success(f"**つねの決裁（承認）**\n\n{verdict.get('raw', '')}")


# ══════════════════════════════════════════════════════════════════════════════
# フェーズ別 UI
# ══════════════════════════════════════════════════════════════════════════════

def _render_phase_a() -> None:
    """フェーズA: テーマ入力と開始ボタン。AT_PENDING が None のときに表示。"""
    # 過去ラウンドを折りたたんで表示
    for past_round in st.session_state.get(SK.AT_HISTORY, []):
        _render_round_result(past_round, collapsed=True)

    st.markdown("---")
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
            use_container_width=True,
        )
    with col2:
        clear_btn = st.button("🗑️ 履歴をリセット", use_container_width=True)
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
                use_container_width=True,
            ):
                clicked_agent = agent

    # つね（意見モード）・決裁ボタン
    col_op, col_verdict, col_reset = st.columns([2, 3, 1])
    with col_op:
        tsune_opinion_btn = st.button(
            f"💬 つねに意見を言わせる",
            key="btn_tsune_opinion",
            use_container_width=True,
        )
    with col_verdict:
        tsune_verdict_btn = st.button(
            f"👔 つねに決裁させる",
            key="btn_tsune_verdict",
            type="primary",
            use_container_width=True,
        )
    with col_reset:
        reset_btn = st.button("🗑️ リセット", key="btn_reset_b", use_container_width=True)

    st.divider()

    # ── スレッド表示（ボタンの下）──────────────────────────────────────────────
    for msg in thread:
        _render_speech_bubble(msg["name"], msg["avatar"], msg["content"])

    # ── ボタン処理（描画後に実行）─────────────────────────────────────────────
    if clicked_agent is not None:
        opinion = run_one_agent(clicked_agent, theme, thread)
        pending["thread"].append(opinion)
        st.session_state[SK.AT_PENDING] = pending
        st.rerun()

    if tsune_opinion_btn:
        tsune_op = run_tsune_opinion(theme, thread)
        pending["thread"].append(tsune_op)
        st.session_state[SK.AT_PENDING] = pending
        st.rerun()

    if tsune_verdict_btn:
        st.divider()
        result = run_tsune_verdict(theme, round_num, thread)
        st.session_state[SK.AT_HISTORY] = st.session_state.get(SK.AT_HISTORY, []) + [result]
        st.session_state[SK.AT_PENDING] = None
        if result["tsune_verdict"]["approved"]:
            save_agent_team_log(result)
            st.balloons()
            st.success("🎉 つねが承認しました！決定事項ログに保存されました。")
            st.info("👇 履歴を開いて「鈴木さんにコードを生成させる」ボタンを押してください。")
        else:
            judgment = result["tsune_verdict"]["judgment"]
            if judgment == "修正":
                st.warning("⚠️ 修正して継続してください。")
            else:
                st.error("❌ 却下されました。テーマや方針を見直してください。")
        st.rerun()

    if reset_btn:
        st.session_state[SK.AT_PENDING] = None
        st.session_state[SK.AT_HISTORY] = []
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

    pending = st.session_state.get(SK.AT_PENDING)

    if pending is None:
        # フェーズA: テーマ入力
        _render_phase_a()
    else:
        # フェーズB: ディスカッション中
        _render_phase_b(pending)


def render_agent_team() -> None:
    """エージェントチーム議論ページのメインエントリポイント。"""
    st.title("🤝 エージェントチーム議論")
    st.caption(
        "4人が自由にディスカッション → 準備ができたら **つね** に決裁を依頼。"
        "つねの承認なしには決定事項に移行できません。"
    )

    engine = st.session_state.get(SK.AI_ENGINE, "ollama")
    st.caption(f"🤖 使用中: {'Gemini API' if engine == 'gemini' else 'Ollama（ローカル）'}")

    tab_discuss, tab_logs = st.tabs(["💬 議論", "📋 決定事項ログ"])

    with tab_discuss:
        _render_discussion_tab()

    with tab_logs:
        logs = load_agent_team_logs()
        _render_log_history(logs)
