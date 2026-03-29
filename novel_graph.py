# -*- coding: utf-8 -*-
"""
novel_graph.py
==============
小説AI「波乱丸」の登場人物・企業間の関係グラフを管理するモジュール。

グラフ理論的位置づけ:
  - ノード = 登場人物（固定エージェント）＋ 登場企業（動的）
  - 有向重み付きエッジ = 関係（type × strength）
  - エピソードをまたぐたびに strength が変化（時系列グラフ）
  - D3.js フォースグラフで可視化（IroFish風）

関係タイプ:
  ally       (+) 同盟・協力
  trust      (+) 信頼
  rival      (-) 対立・競争
  suspicion  (-) 疑惑・不信
  dependence (±) 依存
  neutral        中立（デフォルト）
"""
from __future__ import annotations

import os
import json
import re
import sqlite3

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NOVEL_DB  = os.path.join(_BASE_DIR, "data", "novelist_agent.db")


def _extract_outermost_json(text: str) -> list[dict]:
    """
    テキスト中の最外殻 { ... } ブロックをすべて抽出してパースする。
    非貪欲regexと異なり、ネストされたオブジェクトを正しく処理する。
    """
    results: list[dict] = []
    i = 0
    while i < len(text):
        brace = text.find("{", i)
        if brace == -1:
            break
        depth, j, in_str, escape = 0, brace, False, False
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
                            results.append(json.loads(text[brace:j + 1]))
                        except Exception:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            break
    return results

# ── 固定エージェントノード ──────────────────────────────────────────────────
AGENT_NODES: list[dict] = [
    {"id": "Tune",     "label": "Tune",     "group": "agent", "color": "#3b82f6"},
    {"id": "Dr.Algo",  "label": "Dr.Algo",  "group": "agent", "color": "#ef4444"},
    {"id": "軍師",      "label": "軍師",      "group": "agent", "color": "#f59e0b"},
    {"id": "タム",      "label": "タム",      "group": "agent", "color": "#22c55e"},
    {"id": "リースくん","label": "リースくん","group": "agent", "color": "#8b5cf6"},
]
AGENT_IDS = {a["id"] for a in AGENT_NODES}

# 関係タイプ定義
REL_TYPES = {
    "ally":       {"color": "#22c55e", "label": "同盟"},
    "trust":      {"color": "#3b82f6", "label": "信頼"},
    "rival":      {"color": "#ef4444", "label": "対立"},
    "suspicion":  {"color": "#f97316", "label": "疑惑"},
    "dependence": {"color": "#8b5cf6", "label": "依存"},
    "neutral":    {"color": "#64748b", "label": "中立"},
}

# ── DB初期化 ────────────────────────────────────────────────────────────────

def init_graph_db() -> None:
    conn = sqlite3.connect(_NOVEL_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS novel_relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_no  INTEGER NOT NULL,
            ts          TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            target      TEXT    NOT NULL,
            rel_type    TEXT    NOT NULL DEFAULT 'neutral',
            strength    REAL    NOT NULL DEFAULT 0,
            delta       REAL    NOT NULL DEFAULT 0,
            note        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_novel_rel_ep ON novel_relationships(episode_no);

        CREATE TABLE IF NOT EXISTS civ_characteristics (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT    NOT NULL UNIQUE,
            traits       TEXT,
            goals        TEXT,
            ideology     TEXT,
            strengths    TEXT,
            weaknesses   TEXT,
            personality  TEXT,
            created_at   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationship_predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT    NOT NULL,
            target       TEXT    NOT NULL,
            prediction   TEXT,
            risk_level   REAL    DEFAULT 0.5,
            created_at   TEXT    NOT NULL,
            UNIQUE(source, target)
        );
    """)
    # 初期関係を未登録なら seed する
    existing = conn.execute("SELECT COUNT(*) FROM novel_relationships").fetchone()[0]
    if existing == 0:
        _seed_initial_relations(conn)
    conn.commit()
    conn.close()


def _seed_initial_relations(conn: sqlite3.Connection) -> None:
    """第1話開始時点の関係性（プリセット）"""
    seeds = [
        # (source, target, rel_type, strength, note)
        ("Tune",    "Dr.Algo",   "rival",      -1.0, "データ至上主義 vs 人情派で常に衝突"),
        ("Tune",    "軍師",       "trust",      +2.0, "冷静なTuneが唯一心を許す古参"),
        ("Tune",    "タム",       "dependence", +1.0, "タムの直感が案外核心を突く"),
        ("Tune",    "リースくん", "trust",      +1.5, "真面目な新人への期待"),
        ("Dr.Algo", "軍師",       "rival",      -2.0, "数値 vs 定性で永遠に平行線"),
        ("Dr.Algo", "タム",       "suspicion",  -1.0, "タムの非論理的言動が理解できない"),
        ("軍師",    "タム",       "ally",       +2.0, "孫子とわんわんで謎の共鳴"),
        ("軍師",    "リースくん", "trust",      +1.0, "後継者として目をかけている"),
        ("タム",    "リースくん", "ally",       +1.5, "なんか仲良し（理由不明）"),
    ]
    now = "2000-01-01 00:00:00"
    for s, t, rt, st, note in seeds:
        conn.execute(
            "INSERT INTO novel_relationships (episode_no, ts, source, target, rel_type, strength, delta, note) "
            "VALUES (0, ?, ?, ?, ?, ?, 0, ?)",
            (now, s, t, rt, st, note)
        )


# ── 関係状態の取得（最新エピソードまでの累積） ──────────────────────────────

def get_current_graph(up_to_episode: int | None = None) -> dict:
    """
    最新の関係グラフ状態を返す。
    edges: {(source, target): {rel_type, strength, note, episode_no}}
    """
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    q = "SELECT source, target, rel_type, strength, note, episode_no FROM novel_relationships"
    params: list = []
    if up_to_episode is not None:
        q += " WHERE episode_no <= ?"
        params.append(up_to_episode)
    q += " ORDER BY episode_no ASC, id ASC"
    rows = conn.execute(q, params).fetchall()
    conn.close()

    # 最新エントリで上書き（後勝ち）
    edges: dict[tuple, dict] = {}
    for src, tgt, rt, st, note, ep in rows:
        key = (src, tgt)
        edges[key] = {"rel_type": rt, "strength": st, "note": note or "", "episode_no": ep}

    return edges


def save_relationship_updates(episode_no: int, updates: list[dict]) -> None:
    """
    AIが出力した関係性更新リストをDBに保存する。
    updates = [{"source": str, "target": str, "rel_type": str, "delta": float, "note": str}]
    """
    import datetime
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    current = get_current_graph(up_to_episode=episode_no - 1)

    for u in updates:
        src   = u.get("source", "").strip()
        tgt   = u.get("target", "").strip()
        rt    = u.get("rel_type", "neutral")
        delta = float(u.get("delta", 0))
        note  = u.get("note", "")

        if not src or not tgt:
            continue

        prev = current.get((src, tgt), {})
        prev_strength = prev.get("strength", 0.0)
        new_strength  = max(-5.0, min(5.0, prev_strength + delta))

        conn.execute(
            "INSERT INTO novel_relationships (episode_no, ts, source, target, rel_type, strength, delta, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (episode_no, ts, src, tgt, rt, new_strength, delta, note)
        )

    conn.commit()
    conn.close()


# ── 文明特性 ──────────────────────────────────────────────────────────────────

def get_all_civ_characteristics() -> dict[str, dict]:
    """全企業の特性を返す。{company_name: {traits, goals, ideology, strengths, weaknesses, personality}}"""
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute(
        "SELECT company_name, traits, goals, ideology, strengths, weaknesses, personality "
        "FROM civ_characteristics"
    ).fetchall()
    conn.close()
    result: dict[str, dict] = {}
    for row in rows:
        result[row[0]] = {
            "traits": row[1] or "", "goals": row[2] or "",
            "ideology": row[3] or "", "strengths": row[4] or "",
            "weaknesses": row[5] or "", "personality": row[6] or "",
        }
    return result


def save_civ_characteristics(chars: list[dict]) -> None:
    """文明特性をDBに保存（UPSERT）"""
    import datetime
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in chars:
        name = c.get("company_name", "").strip()
        if not name:
            continue
        conn.execute(
            """INSERT INTO civ_characteristics
               (company_name, traits, goals, ideology, strengths, weaknesses, personality, created_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(company_name) DO UPDATE SET
               traits=excluded.traits, goals=excluded.goals, ideology=excluded.ideology,
               strengths=excluded.strengths, weaknesses=excluded.weaknesses,
               personality=excluded.personality, created_at=excluded.created_at""",
            (name, c.get("traits",""), c.get("goals",""), c.get("ideology",""),
             c.get("strengths",""), c.get("weaknesses",""), c.get("personality",""), ts)
        )
    conn.commit()
    conn.close()


def generate_civ_characteristics_ai() -> int:
    """
    文明レジストリの企業リストをGeminiに渡して個性・目標・思想を生成し保存する。
    Returns: 生成した企業数
    """
    try:
        from novelist_agent import get_civilization_registry
        civs = get_civilization_registry()
    except Exception:
        return 0

    _skip = {"(企業名は不明)", "(不明)", "(システム上の記録なし)"}
    valid_civs = [c for c in civs if c.get("company_name") and c["company_name"] not in _skip]
    if not valid_civs:
        return 0

    # 既に特性がある企業は除外（スキップ）
    existing = get_all_civ_characteristics()
    new_civs = [c for c in valid_civs if c["company_name"] not in existing]
    if not new_civs:
        return 0

    civ_list = "\n".join(
        f"・{c['company_name']}（{c['industry']} / {c.get('civ_era','?')} / status:{c.get('status','active')}）"
        for c in new_civs
    )

    prompt = f"""以下の企業・文明リストについて、それぞれの個性・目標・思想をSFドラマ世界観（宇宙時代のリース審査）で創造してください。
各企業は自律的に動く存在として設計し、次のエピソードで文豪AIがストーリーを生成する際に参照します。

【企業・文明リスト】
{civ_list}

【出力形式（JSONのみ、説明不要）】
```json
{{"civ_characteristics": [
  {{
    "company_name": "企業名（リストと完全一致）",
    "traits": "個性・特徴（1〜2文）",
    "goals": "目標・野望（1〜2文）",
    "ideology": "思想・価値観（1〜2文）",
    "strengths": "強み（キーワード3つ程度）",
    "weaknesses": "弱み・脆弱性（キーワード3つ程度）",
    "personality": "性格キーワード（例：好戦的・神秘的・合理主義）"
  }}
]}}
```
すべての企業について出力すること。"""

    try:
        from ai_chat import _chat_for_thread, _get_gemini_key_from_secrets, GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT
        api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()
        if not api_key:
            return 0
        raw = _chat_for_thread(
            "gemini", "", [{"role": "user", "content": prompt}],
            timeout_seconds=90, api_key=api_key, gemini_model=GEMINI_MODEL_DEFAULT
        )
        text = (raw.get("message") or {}).get("content", "") or ""
    except Exception:
        return 0

    chars = []
    for data in _extract_outermost_json(text):
        if "civ_characteristics" in data:
            chars.extend(data["civ_characteristics"])

    if not chars:
        return 0

    save_civ_characteristics(chars)
    return len(chars)


# ── 関係性未来予測 ──────────────────────────────────────────────────────────

def get_all_relationship_predictions() -> dict[tuple, dict]:
    """全関係の未来予測を返す。{(source, target): {prediction, risk_level}}"""
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute(
        "SELECT source, target, prediction, risk_level FROM relationship_predictions"
    ).fetchall()
    conn.close()
    return {(r[0], r[1]): {"prediction": r[2] or "", "risk_level": r[3] or 0.5}
            for r in rows}


def save_relationship_predictions(preds: list[dict]) -> None:
    """関係予測をDBに保存（UPSERT）"""
    import datetime
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in preds:
        src = p.get("source", "").strip()
        tgt = p.get("target", "").strip()
        if not src or not tgt:
            continue
        conn.execute(
            """INSERT INTO relationship_predictions (source, target, prediction, risk_level, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(source, target) DO UPDATE SET
               prediction=excluded.prediction, risk_level=excluded.risk_level,
               created_at=excluded.created_at""",
            (src, tgt, p.get("prediction",""), float(p.get("risk_level", 0.5)), ts)
        )
    conn.commit()
    conn.close()


def generate_relationship_predictions_ai() -> int:
    """
    現在の関係グラフとキャラクター特性を元にGeminiが未来を予測する。
    Returns: 予測したエッジ数
    """
    edges = get_current_graph()
    chars = get_all_civ_characteristics()

    if not edges:
        return 0

    # 関係テキスト構築
    rel_lines = []
    for (src, tgt), info in list(edges.items())[:30]:  # 多すぎるとトークン超過
        st = info["strength"]
        note = info["note"]
        rel_lines.append(f"・{src} → {tgt}: {info['rel_type']} [{st:+.1f}]  {note}")
    rel_text = "\n".join(rel_lines)

    # 特性テキスト構築
    char_lines = []
    for name, c in list(chars.items())[:15]:
        char_lines.append(f"・{name}: {c['traits']} | 目標:{c['goals']} | 性格:{c['personality']}")
    char_text = "\n".join(char_lines) if char_lines else "（特性未生成）"

    prompt = f"""以下の登場人物・企業間の現在の関係グラフを分析し、今後の展開を予測してください。
各関係がどの方向に進化するか、どんな事件・衝突・同盟が起きそうかを具体的に予測します。

【現在の関係グラフ】
{rel_text}

【各文明の特性】
{char_text}

【出力形式（JSONのみ、説明不要）】
```json
{{"relationship_predictions": [
  {{
    "source": "登場人物/企業名A",
    "target": "登場人物/企業名B",
    "prediction": "今後の予測（1〜2文。具体的なイベント・ターニングポイントを含む）",
    "risk_level": 0.0〜1.0の数値（0=安定・1=壊滅的変動）
  }}
]}}
```
すべての関係について予測すること。risk_level は小数点1桁で。"""

    try:
        from ai_chat import _chat_for_thread, _get_gemini_key_from_secrets, GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT
        api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()
        if not api_key:
            return 0
        raw = _chat_for_thread(
            "gemini", "", [{"role": "user", "content": prompt}],
            timeout_seconds=90, api_key=api_key, gemini_model=GEMINI_MODEL_DEFAULT
        )
        text = (raw.get("message") or {}).get("content", "") or ""
    except Exception:
        return 0

    preds = []
    for data in _extract_outermost_json(text):
        if "relationship_predictions" in data:
            preds.extend(data["relationship_predictions"])

    if not preds:
        return 0

    save_relationship_predictions(preds)
    return len(preds)


# ── 企業間関係の自動想像・シード ─────────────────────────────────────────────

def generate_and_seed_company_relations() -> int:
    """
    文明レジストリの企業リストをGeminiに渡して企業間関係を想像させ、
    novel_relationships DB にシード（episode_no=-1）する。
    既に企業間エッジが存在する場合は追加しない。
    Returns: 追加したエッジ数
    """
    import datetime

    # 既存の企業間エッジ確認
    existing = get_current_graph()
    company_company = [(k, v) for k, v in existing.items()
                       if k[0] not in AGENT_IDS and k[1] not in AGENT_IDS and not v.get("auto")]
    if company_company:
        return 0  # 既にある

    # 文明レジストリから有効な企業名を取得（重複排除・不明除外）
    try:
        from novelist_agent import get_civilization_registry
        civs = get_civilization_registry()
    except Exception:
        return 0

    _skip = {"(企業名は不明)", "(不明)", "(システム上の記録なし)"}
    seen: set[str] = set()
    valid_civs = []
    for c in civs:
        name = c.get("company_name", "")
        if name and name not in _skip and name not in seen:
            seen.add(name)
            valid_civs.append(c)

    if len(valid_civs) < 2:
        return 0

    # Gemini に企業間関係を想像させる
    civ_list = "\n".join(
        f"・{c['company_name']}（{c['industry']} / {c.get('civ_era','?')}）"
        for c in valid_civs
    )
    prompt = f"""以下の企業・文明リストを見て、それぞれの間にどんな関係があるか想像して物語的に設定してください。
宇宙開発競争、資源争奪、同盟、情報戦、教育格差、技術提携など自由に創造してください。

【企業・文明リスト】
{civ_list}

【出力形式（JSONのみ、説明不要）】
```json
{{"relationship_updates": [
  {{"source": "企業名A", "target": "企業名B", "rel_type": "rival", "delta": -3, "note": "宇宙開発権をめぐり激しく対立"}},
  {{"source": "企業名C", "target": "企業名D", "rel_type": "ally", "delta": 2, "note": "共同で銀河航路を開拓"}},
  ...
]}}
```
rel_type は ally / trust / rival / suspicion / dependence / neutral のいずれか。
delta は -5〜+5。すべての企業ペアに関係を作る必要はないが、ドラマチックな関係を10〜15個程度作ること。"""

    try:
        from ai_chat import _chat_for_thread, _get_gemini_key_from_secrets, GEMINI_API_KEY_ENV, GEMINI_MODEL_DEFAULT
        api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()
        if not api_key:
            return 0
        raw = _chat_for_thread(
            "gemini", "", [{"role": "user", "content": prompt}],
            timeout_seconds=60, api_key=api_key, gemini_model=GEMINI_MODEL_DEFAULT
        )
        text = (raw.get("message") or {}).get("content", "") or ""
    except Exception:
        return 0

    updates = parse_relationship_updates_from_novel(text)
    if not updates:
        return 0

    # episode_no = -1 としてシード（初期世界設定）
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    count = 0
    for u in updates:
        src = u.get("source", "").strip()
        tgt = u.get("target", "").strip()
        rt  = u.get("rel_type", "neutral")
        delta = float(u.get("delta", 0))
        note  = u.get("note", "")
        if not src or not tgt or src in AGENT_IDS or tgt in AGENT_IDS:
            continue
        st = max(-5.0, min(5.0, delta))
        conn.execute(
            "INSERT INTO novel_relationships (episode_no, ts, source, target, rel_type, strength, delta, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (-1, ts, src, tgt, rt, st, delta, note)
        )
        count += 1
    conn.commit()
    conn.close()
    return count


# ── AIプロンプト用テキスト生成 ───────────────────────────────────────────────

def build_graph_context_for_prompt(episode_no: int) -> str:
    """
    直前エピソードまでの関係グラフ状態、文明特性、未来予測をテキスト化してプロンプトに注入する。
    """
    edges = get_current_graph(up_to_episode=episode_no - 1)
    characteristics = get_all_civ_characteristics()
    predictions = get_all_relationship_predictions()

    if not edges and not characteristics:
        return ""

    lines = [f"\n【登場人物・企業の関係グラフ（第{episode_no - 1}話終了時点）】",
             f"以下の関係性・特性・予測を必ず第{episode_no}話の展開・セリフ・心理描写に反映すること。",
             ""]

    # 文明特性（企業の自律的行動の根拠）
    if characteristics:
        lines.append("▼ 各文明・企業の特性（自律的行動の根拠として使用すること）")
        for name, c in list(characteristics.items())[:12]:
            traits = c.get("traits", "")
            goals  = c.get("goals", "")
            pers   = c.get("personality", "")
            weak   = c.get("weaknesses", "")
            info_parts = []
            if traits: info_parts.append(f"特徴:{traits}")
            if goals:  info_parts.append(f"目標:{goals}")
            if pers:   info_parts.append(f"性格:{pers}")
            if weak:   info_parts.append(f"弱点:{weak}")
            if info_parts:
                lines.append(f"  【{name}】" + " / ".join(info_parts))
        lines.append("")

    if edges:
        # エージェント間
        agent_edges = {k: v for k, v in edges.items() if k[0] in AGENT_IDS and k[1] in AGENT_IDS}
        if agent_edges:
            lines.append("▼ エージェント間（この関係を必ずセリフ・態度・心理に反映すること）")
            for (src, tgt), info in sorted(agent_edges.items()):
                rt_label = REL_TYPES.get(info["rel_type"], {}).get("label", info["rel_type"])
                st = info["strength"]
                bar = "█" * int(abs(st)) + "░" * (5 - int(abs(st)))
                sign = "+" if st >= 0 else ""
                pred = predictions.get((src, tgt)) or predictions.get((tgt, src)) or {}
                pred_text = f" → 予測: {pred['prediction']}" if pred.get("prediction") else ""
                lines.append(f"  {src} → {tgt}: {rt_label} [{sign}{st:.1f}] {bar}  ※{info['note']}{pred_text}")

        # 企業・文明間の関係
        company_edges = {k: v for k, v in edges.items() if k[0] not in AGENT_IDS or k[1] not in AGENT_IDS}
        if company_edges:
            lines.append("")
            lines.append("▼ 企業・文明との関係と未来予測（これらを物語に必ず織り込むこと）")
            for (src, tgt), info in sorted(company_edges.items()):
                rt_label = REL_TYPES.get(info["rel_type"], {}).get("label", info["rel_type"])
                st = info["strength"]
                sign = "+" if st >= 0 else ""
                note = f"  ※{info['note']}" if info["note"] else ""
                pred = predictions.get((src, tgt)) or predictions.get((tgt, src)) or {}
                risk = pred.get("risk_level", 0)
                risk_text = ""
                if pred.get("prediction"):
                    risk_emoji = "🔴" if risk >= 0.7 else "🟡" if risk >= 0.4 else "🟢"
                    risk_text = f"  {risk_emoji}予測: {pred['prediction']}"
                lines.append(f"  {src} → {tgt}: {rt_label} [{sign}{st:.1f}]{note}{risk_text}")

    lines.append("")
    lines.append("【重要】上記の特性・予測を踏まえ、各文明が自律的な意思と目標を持って行動する物語を描くこと。")
    lines.append("予測されたリスクや衝突を第話の展開の核心として使用し、驚きのある展開を作ること。")

    return "\n".join(lines)


# ── AIが出力したJSONを解析 ───────────────────────────────────────────────────

def parse_relationship_updates_from_novel(novel_body: str) -> list[dict]:
    """
    小説本文末尾の ```json ... ``` ブロックから関係性更新を抽出する。
    ネストされたJSONも正しく処理する。
    """
    updates = []
    for data in _extract_outermost_json(novel_body):
        if "relationship_updates" in data:
            updates.extend(data["relationship_updates"])
    return updates


# ── D3.js 可視化用データ構築 ──────────────────────────────────────────────────

def build_d3_graph_data(episode_no: int | None = None) -> dict:
    """
    D3.js フォースグラフ用の nodes / links データを返す。
    novel_relationships エッジに加え、文明レジストリの企業も自動的にノードとして追加する。
    文明特性・関係予測も含む。
    """
    edges = get_current_graph(up_to_episode=episode_no)
    characteristics = get_all_civ_characteristics()
    predictions = get_all_relationship_predictions()

    # ノード収集（エッジから）
    node_ids: set[str] = set(AGENT_IDS)
    for (src, tgt) in edges.keys():
        node_ids.add(src)
        node_ids.add(tgt)

    agent_map = {a["id"]: a for a in AGENT_NODES}
    nodes = []
    for nid in node_ids:
        if nid in agent_map:
            nodes.append({**agent_map[nid], "size": 20})
        else:
            char = characteristics.get(nid, {})
            nodes.append({"id": nid, "label": nid[:12], "group": "company",
                          "color": "#94a3b8", "size": 14,
                          "traits": char.get("traits", ""),
                          "goals": char.get("goals", ""),
                          "ideology": char.get("ideology", ""),
                          "personality": char.get("personality", "")})

    # 文明レジストリから企業ノードを追加（エッジなしでも表示）
    # 審査結果に応じて色分け・Tuneへのエッジ自動生成
    _STATUS_COLOR = {
        "active":    "#22c55e",   # 緑 — 活動中
        "collapsed": "#ef4444",   # 赤 — 滅亡
        "ascended":  "#f59e0b",   # 金 — 昇華
        "dormant":   "#64748b",   # グレー — 休眠
    }
    # auto エッジの色（承認=青、否決/滅亡=赤、昇華=金、休眠=グレー）
    _STATUS_AUTO_COLOR = {
        "active":    "#3b82f6",   # 青 — 審査通過
        "collapsed": "#ef4444",   # 赤 — 審査否決/滅亡
        "ascended":  "#f59e0b",   # 金 — 昇華
        "dormant":   "#64748b",   # グレー — 休眠
    }
    _STATUS_TO_REL = {
        "active":    ("trust",   +2.0, "審査通過"),
        "collapsed": ("rival",   -3.0, "審査否決"),
        "ascended":  ("ally",    +4.0, "審査通過"),
        "dormant":   ("neutral",  0.0, ""),
    }
    try:
        from novelist_agent import get_civilization_registry as _get_civs
        civs = _get_civs()
    except Exception:
        civs = []

    existing_ids = {n["id"] for n in nodes}
    auto_links: list[dict] = []

    for civ in civs:
        cid = civ["company_name"]  # 企業名をノードIDとして使用
        if not cid:
            continue
        # エピソードフィルタ
        if episode_no is not None and civ.get("first_episode", 0) > episode_no:
            continue
        status = civ.get("status", "active")
        color = _STATUS_COLOR.get(status, "#94a3b8")
        industry = civ.get("industry", "")
        label = cid[:12]

        if cid not in existing_ids:
            char = characteristics.get(cid, {})
            nodes.append({
                "id": cid,
                "label": label,
                "group": "company",
                "color": color,
                "size": 14,
                "industry": industry,
                "status": status,
                "traits": char.get("traits", ""),
                "goals": char.get("goals", ""),
                "ideology": char.get("ideology", ""),
                "personality": char.get("personality", ""),
            })
            existing_ids.add(cid)
        else:
            # 既存ノードの色を審査結果に合わせて更新
            for n in nodes:
                if n["id"] == cid:
                    n["color"] = color
                    break

        # Tune → 企業 の自動エッジ（novel_relationshipsに同ペアがなければ）
        if ("Tune", cid) not in edges and (cid, "Tune") not in edges:
            rt, st, note = _STATUS_TO_REL.get(status, ("neutral", 0.0, ""))
            ep = civ.get("first_episode", 0)
            auto_color = _STATUS_AUTO_COLOR.get(status, "#64748b")
            auto_links.append({
                "source": "Tune",
                "target": cid,
                "rel_type": rt,
                "rel_label": note,   # "審査通過" / "審査否決" を表示
                "strength": st,
                "width": 1.5,
                "color": auto_color,
                "opacity": 0.5,
                "note": note,
                "episode_no": ep,
                "auto": True,
            })

    # エッジ（novel_relationships から）
    links = []
    for (src, tgt), info in edges.items():
        st = info["strength"]
        rt = info["rel_type"]
        color = REL_TYPES.get(rt, {}).get("color", "#64748b")
        pred = predictions.get((src, tgt)) or predictions.get((tgt, src)) or {}
        links.append({
            "source": src,
            "target": tgt,
            "rel_type": rt,
            "rel_label": REL_TYPES.get(rt, {}).get("label", rt),
            "strength": st,
            "width": max(1.0, abs(st) * 0.8 + 1),
            "color": color,
            "opacity": 0.3 + min(0.6, abs(st) / 5 * 0.6),
            "note": info["note"],
            "episode_no": info["episode_no"],
            "auto": False,
            "prediction": pred.get("prediction", ""),
            "risk_level": pred.get("risk_level", 0.0),
        })

    links.extend(auto_links)

    return {"nodes": nodes, "links": links}


def get_episode_history(source: str, target: str) -> list[dict]:
    """特定エッジの時系列変化を返す（グラフクリック時の詳細表示用）"""
    init_graph_db()
    conn = sqlite3.connect(_NOVEL_DB)
    rows = conn.execute(
        "SELECT episode_no, rel_type, strength, delta, note FROM novel_relationships "
        "WHERE source=? AND target=? ORDER BY episode_no ASC",
        (source, target)
    ).fetchall()
    conn.close()
    return [{"ep": r[0], "type": r[1], "strength": r[2], "delta": r[3], "note": r[4]}
            for r in rows]
