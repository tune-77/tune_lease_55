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

def _fix_json_str(raw: str) -> str:
    """
    Geminiが出力しがちな不正JSON文字列を修正する。
    - 文字列値内の生改行 → \\n
    - 文字列値内のタブ → \\t
    """
    result = []
    in_str = False
    escape = False
    for c in raw:
        if escape:
            escape = False
            result.append(c)
        elif c == "\\":
            escape = True
            result.append(c)
        elif c == '"':
            in_str = not in_str
            result.append(c)
        elif in_str and c == "\n":
            result.append("\\n")
        elif in_str and c == "\r":
            pass  # 除去
        elif in_str and c == "\t":
            result.append("\\t")
        else:
            result.append(c)
    return "".join(result)


def _build_stack(s: str) -> tuple[list[str], bool]:
    """文字列を走査してネストスタックと in_str フラグを返す"""
    stack: list[str] = []
    in_str = False
    escape = False
    for c in s:
        if escape:             escape = False
        elif c == "\\":        escape = True
        elif c == '"':         in_str = not in_str
        elif not in_str:
            if c in "{[":      stack.append(c)
            elif c == "}" and stack and stack[-1] == "{": stack.pop()
            elif c == "]" and stack and stack[-1] == "[": stack.pop()
    return stack, in_str


def _close_stack(s: str, stack: list[str]) -> str:
    """末尾カンマ除去 → スタック逆順で閉じる"""
    s = s.rstrip()
    while s and s[-1] in ",: \t":
        s = s[:-1].rstrip()
    for open_char in reversed(stack):
        s += "}" if open_char == "{" else "]"
    return s


def _repair_truncated_json(s: str) -> str:
    """
    途中で切れたJSONを修復する。
    ① 文字列の途中で切断 → 開き引用符まで逆行して安全位置に切り詰め
    ② 末尾のカンマ・コロン除去
    ③ スタック逆順で正確な閉じ順（]}]}）補完
    """
    s = _fix_json_str(s).rstrip()
    stack, in_str = _build_stack(s)

    if in_str:
        # 開き引用符まで逆行
        j = len(s) - 1
        while j >= 0:
            if s[j] == '"':
                # エスケープされていないか確認
                num_bs = 0
                k = j - 1
                while k >= 0 and s[k] == '\\':
                    num_bs += 1
                    k -= 1
                if num_bs % 2 == 0:   # 非エスケープ引用符 → 開き引用符
                    break
            j -= 1
        # 開き引用符より前に切り詰め、スタック再計算
        s = s[:j]
        stack, _ = _build_stack(s)

    return _close_stack(s, stack)


def _parse_simulation_json(text: str) -> dict:
    """
    AIレスポンスから events キーを持つ JSON を多段フォールバックで抽出する。
    戦略1: ```json...``` ブロック内テキストを直接 json.loads
    戦略2: 上記テキストを _fix_json_str で修正後 json.loads
    戦略3: 全文から最初の { を起点に括弧深度カウントで抽出→修正→parse
    """
    def _try_parse(s: str) -> dict | None:
        for attempt in [s, _fix_json_str(s), _repair_truncated_json(s)]:
            try:
                d = json.loads(attempt)
                if "events" in d:
                    return d
            except Exception:
                pass
        return None

    # 戦略1・2: ```json...``` を探してその中を解析
    pos = 0
    while True:
        start = text.find("```json", pos)
        if start == -1:
            start = text.find("```", pos)
            if start == -1:
                break
            json_text_start = start + 3
        else:
            json_text_start = start + 7
        end = text.find("```", json_text_start)
        block = text[json_text_start:end].strip() if end != -1 else text[json_text_start:].strip()
        d = _try_parse(block)
        if d:
            return d
        pos = json_text_start + 1
        if end == -1:
            break

    # 戦略3: テキスト全体から括弧深度カウントで抽出
    i = 0
    while i < len(text):
        brace = text.find("{", i)
        if brace == -1:
            break
        depth, j, in_s, esc = 0, brace, False, False
        while j < len(text):
            c = text[j]
            if esc:              esc = False
            elif c == "\\":     esc = True
            elif c == '"':      in_s = not in_s
            elif not in_s:
                if c == "{":    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        d = _try_parse(text[brace:j + 1])
                        if d:
                            return d
                        i = j + 1
                        break
            j += 1
        else:
            break
    return {}


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
【ルール】
・各文明は特性・目標に従って自律行動する
・予測リスクは確率的に現実化させる
・イベントは3〜5個（多すぎない）
・descriptionは1〜2文で簡潔に
・関係変化はdeltaで反映（-5〜+5）

【出力形式：JSONのみ・説明なし】
```json
{{
  "round_no": {next_round},
  "year": {year},
  "events": [
    {{"civ": "文明名", "event_type": "war|alliance|collapse|discovery|growth|betrayal|revolution|contact", "title": "タイトル20字以内", "description": "1〜2文", "affected": ["関連文明名"]}}
  ],
  "relationship_updates": [
    {{"source": "A", "target": "B", "rel_type": "rival", "delta": -2.0, "note": "変化理由"}}
  ],
  "summary": "ラウンド総括（2文）"
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

    # JSONパース（多段フォールバック）
    result_data = _parse_simulation_json(text)
    if not result_data:
        # 詳細診断: 最初の{からjson.loadsのエラー内容を取得
        _diag = ""
        _brace = text.find("{")
        if _brace != -1:
            try:
                json.loads(_fix_json_str(text[_brace:]))
            except json.JSONDecodeError as _je:
                _diag = f" | JSONエラー: {_je}"
        return {"error": f"AIの応答を解析できませんでした: {text[:200]}{_diag}"}

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
