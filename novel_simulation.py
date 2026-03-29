# -*- coding: utf-8 -*-
"""
novel_simulation.py
===================
MiroFish風 宇宙文明シミュレーションエンジン。
1ラウンド = 100年。ラウンドごとにGeminiが各文明の行動を自律シミュレートし、
関係グラフを更新・コメントを生成する。
"""
from __future__ import annotations

import os
import json
import re
import sqlite3
import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NOVEL_DB  = os.path.join(_BASE_DIR, "data", "novelist_agent.db")

YEARS_PER_ROUND = 100

# イベントタイプ定義
EVENT_TYPES = {
    "war":        {"emoji": "⚔️",  "color": "#ef4444", "label": "戦争"},
    "alliance":   {"emoji": "🤝",  "color": "#22c55e", "label": "同盟"},
    "collapse":   {"emoji": "💀",  "color": "#64748b", "label": "崩壊"},
    "discovery":  {"emoji": "🔭",  "color": "#38bdf8", "label": "発見"},
    "growth":     {"emoji": "📈",  "color": "#86efac", "label": "成長"},
    "betrayal":   {"emoji": "🗡️",  "color": "#f97316", "label": "裏切り"},
    "revolution": {"emoji": "⚡",  "color": "#a78bfa", "label": "革命"},
    "contact":    {"emoji": "📡",  "color": "#fde68a", "label": "接触"},
}


def init_simulation_db() -> None:
    conn = sqlite3.connect(_NOVEL_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulation_rounds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            round_no    INTEGER NOT NULL UNIQUE,
            year        INTEGER NOT NULL,
            events      TEXT,
            summary     TEXT,
            created_at  TEXT    NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def get_current_round() -> int:
    """現在のラウンド数（0=未開始）を返す"""
    init_simulation_db()
    conn = sqlite3.connect(_NOVEL_DB)
    row = conn.execute("SELECT MAX(round_no) FROM simulation_rounds").fetchone()
    conn.close()
    return row[0] if row[0] is not None else 0


def get_current_year() -> int:
    """現在のシミュレーション年（ラウンド × 100）を返す"""
    return get_current_round() * YEARS_PER_ROUND


def get_round_history(limit: int = 20) -> list[dict]:
    """ラウンド履歴を新しい順で返す"""
    init_simulation_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute(
        "SELECT round_no, year, events, summary, created_at FROM simulation_rounds "
        "ORDER BY round_no DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        events = []
        try:
            events = json.loads(r[2]) if r[2] else []
        except Exception:
            pass
        result.append({
            "round_no": r[0], "year": r[1], "events": events,
            "summary": r[3] or "", "created_at": r[4]
        })
    return result


def run_simulation_round() -> dict:
    """
    1ラウンド進行する。各文明が自律的に行動し、関係グラフが更新される。
    Returns: ラウンド結果 dict（events, summary, relationship_updates を含む）
             エラー時は {"error": "..."} を返す
    """
    from novel_graph import (
        get_current_graph, get_all_civ_characteristics,
        get_all_relationship_predictions, save_relationship_updates,
        AGENT_IDS,
    )

    try:
        from novelist_agent import get_civilization_registry
        civs = get_civilization_registry()
    except Exception:
        civs = []

    current_round = get_current_round()
    next_round    = current_round + 1
    year          = next_round * YEARS_PER_ROUND

    # 現在の状態を収集
    edges = get_current_graph()
    chars = get_all_civ_characteristics()
    preds = get_all_relationship_predictions()

    # 有効な文明リスト
    _skip = {"(企業名は不明)", "(不明)", "(システム上の記録なし)"}
    valid_civs = [c for c in civs if c.get("company_name") and c["company_name"] not in _skip]

    if not valid_civs:
        return {"error": "文明データがありません。先に文豪AIで小説を生成してください。"}

    # 関係テキスト
    rel_lines = []
    for (src, tgt), info in list(edges.items())[:24]:
        note = info.get("note", "")
        rel_lines.append(
            f"  {src} → {tgt}: {info['rel_type']} [{info['strength']:+.1f}] {note}"
        )

    # 特性テキスト
    char_lines = []
    for name, c in list(chars.items())[:14]:
        parts = []
        if c.get("traits"):    parts.append(f"特徴:{c['traits']}")
        if c.get("goals"):     parts.append(f"目標:{c['goals']}")
        if c.get("personality"): parts.append(f"性格:{c['personality']}")
        if c.get("weaknesses"):  parts.append(f"弱点:{c['weaknesses']}")
        char_lines.append(f"  [{name}] " + " / ".join(parts))

    # 高リスク予測テキスト
    pred_lines = []
    sorted_preds = sorted(preds.items(), key=lambda x: x[1].get("risk_level", 0), reverse=True)
    for (src, tgt), p in sorted_preds[:10]:
        if p.get("risk_level", 0) >= 0.35 and p.get("prediction"):
            pred_lines.append(f"  {src}→{tgt} (risk:{p['risk_level']:.1f}): {p['prediction']}")

    # 前回ラウンドサマリー
    history = get_round_history(limit=2)
    prev_summary = ""
    if len(history) >= 1:
        last = history[0]  # 最新（現在のラウンド）
        if last["round_no"] == current_round and last.get("summary"):
            prev_summary = f"\n【前回（第{current_round}ラウンド / アルカイア暦A.{current_round * YEARS_PER_ROUND}）の出来事】\n{last['summary']}\n"

    civ_list_text = "\n".join(
        f"  ・{c['company_name']}（{c['industry']} / {c.get('status','active')} / "
        f"era:{c.get('civ_era','?')}）"
        for c in valid_civs
    )

    prompt = f"""あなたは宇宙文明シミュレーター「MiroFish」です。
リース審査AIたちが監視する宇宙文明の100年間（第{next_round}ラウンド・アルカイア暦{year}年）をシミュレートしてください。

各文明は自律的に行動します。特性・目標・弱点に基づいて動き、予測リスクが現実となる場合があります。

【現存文明リスト】
{civ_list_text}

【現在の関係グラフ】
{"（関係なし）" if not rel_lines else chr(10).join(rel_lines)}

【各文明の特性・弱点・野望】
{"（特性未生成）" if not char_lines else chr(10).join(char_lines)}

【予測されていたリスク（今回発動の可能性あり）】
{"（予測なし）" if not pred_lines else chr(10).join(pred_lines)}
{prev_summary}
【シミュレーションルール】
・各文明は特性・目標に従って自律行動する（予測リスクは確率的に現実化）
・ドラマチックな転換点を演出（戦争、裏切り、技術革命、神秘的接触など）
・崩壊する文明は容赦なく描写すること
・最低4〜8個のイベントを発生させる
・関係の変化はdeltaで必ず反映（-5〜+5）

【出力形式（JSONのみ、前後に説明不要）】
```json
{{
  "round_no": {next_round},
  "year": {year},
  "events": [
    {{
      "civ": "主体となる文明名",
      "event_type": "war|alliance|collapse|discovery|growth|betrayal|revolution|contact",
      "title": "イベントタイトル（20字以内）",
      "description": "詳細説明（2〜3文。具体的・ドラマチック・文学的に）",
      "affected": ["影響を受ける文明名"]
    }}
  ],
  "relationship_updates": [
    {{"source": "A", "target": "B", "rel_type": "rival", "delta": -2.5, "note": "具体的な変化の理由"}}
  ],
  "summary": "このラウンドの総括（3〜4文。文明の盛衰・時代の変化を文学的に描写）"
}}
```"""

    try:
        from ai_chat import (
            _chat_for_thread, _get_gemini_key_from_secrets,
            GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT,
        )
        api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()
        if not api_key:
            return {"error": "Gemini APIキーが設定されていません"}
        raw = _chat_for_thread(
            "gemini", "",
            [{"role": "user", "content": prompt}],
            timeout_seconds=120, api_key=api_key, gemini_model=GEMINI_MODEL_DEFAULT,
        )
        text = (raw.get("message") or {}).get("content", "") or ""
    except Exception as e:
        return {"error": f"AI通信エラー: {e}"}

    # JSONパース（```json...``` ブロックを正しく抽出）
    result_data: dict = {}

    def _extract_json_blocks(src: str) -> list[str]:
        """```json ... ``` の中身を正しく抽出（ネスト対応）"""
        blocks: list[str] = []
        i = 0
        while True:
            start = src.find("```json", i)
            if start == -1:
                break
            brace_start = src.find("{", start)
            if brace_start == -1:
                break
            depth = 0
            j = brace_start
            in_str = False
            escape = False
            while j < len(src):
                c = src[j]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            blocks.append(src[brace_start:j + 1])
                            i = j + 1
                            break
                j += 1
            else:
                break
        return blocks

    for m in _extract_json_blocks(text):
        try:
            data = json.loads(m)
            if "events" in data:
                result_data = data
                break
        except Exception:
            pass

    if not result_data:
        # フォールバック: テキスト全体から { ... } を抽出
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            j = brace_start
            in_str = False
            escape = False
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = json.loads(text[brace_start:j + 1])
                                if "events" in data:
                                    result_data = data
                            except Exception:
                                pass
                            break
                j += 1

    if not result_data:
        return {"error": f"AIの応答を解析できませんでした: {text[:300]}"}

    # ── DB保存 ────────────────────────────────────────────────────────
    init_simulation_db()
    conn = sqlite3.connect(_NOVEL_DB)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT OR REPLACE INTO simulation_rounds (round_no, year, events, summary, created_at) "
        "VALUES (?,?,?,?,?)",
        (next_round, year,
         json.dumps(result_data.get("events", []), ensure_ascii=False),
         result_data.get("summary", ""),
         ts)
    )
    conn.commit()
    conn.close()

    # ── 関係性更新 ────────────────────────────────────────────────────
    rel_updates = result_data.get("relationship_updates", [])
    if rel_updates:
        # シミュレーション専用エピソード番号: 10000 + round_no
        sim_episode = 10000 + next_round
        save_relationship_updates(sim_episode, rel_updates)

    return result_data
