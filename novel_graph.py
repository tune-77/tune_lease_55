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


# ── AIプロンプト用テキスト生成 ───────────────────────────────────────────────

def build_graph_context_for_prompt(episode_no: int) -> str:
    """
    直前エピソードまでの関係グラフ状態をテキスト化してプロンプトに注入する。
    """
    edges = get_current_graph(up_to_episode=episode_no - 1)
    if not edges:
        return ""

    lines = [f"\n【登場人物・企業の関係グラフ（第{episode_no - 1}話終了時点）】",
             f"以下の関係性を必ず第{episode_no}話の展開・セリフ・心理描写に反映すること。",
             "strength は -5（最大対立）〜 +5（最大連帯）。",
             ""]

    # エージェント間
    agent_edges = {k: v for k, v in edges.items() if k[0] in AGENT_IDS and k[1] in AGENT_IDS}
    if agent_edges:
        lines.append("▼ エージェント間")
        for (src, tgt), info in sorted(agent_edges.items()):
            rt_label = REL_TYPES.get(info["rel_type"], {}).get("label", info["rel_type"])
            st = info["strength"]
            bar = "█" * int(abs(st)) + "░" * (5 - int(abs(st)))
            sign = "+" if st >= 0 else ""
            lines.append(f"  {src} → {tgt}: {rt_label} [{sign}{st:.1f}] {bar}  ※{info['note']}")

    # 企業との関係
    company_edges = {k: v for k, v in edges.items() if k[0] not in AGENT_IDS or k[1] not in AGENT_IDS}
    if company_edges:
        lines.append("")
        lines.append("▼ 企業（文明）との関係")
        for (src, tgt), info in sorted(company_edges.items()):
            rt_label = REL_TYPES.get(info["rel_type"], {}).get("label", info["rel_type"])
            st = info["strength"]
            sign = "+" if st >= 0 else ""
            lines.append(f"  {src} → {tgt}: {rt_label} [{sign}{st:.1f}]  ※{info['note']}")

    lines.append("")
    lines.append("【関係性更新ルール】")
    lines.append("今話の展開で関係が変化した場合、小説末尾に以下の形式で出力せよ（変化がない場合は省略可）：")
    lines.append("```json")
    lines.append('{"relationship_updates": [')
    lines.append('  {"source": "Tune", "target": "Dr.Algo", "rel_type": "rival", "delta": -1, "note": "今回も激しく対立"},')
    lines.append('  {"source": "軍師", "target": "株式会社北斗鉄工", "rel_type": "ally", "delta": 2, "note": "承認を勝ち取った"}')
    lines.append(']}')
    lines.append("```")

    return "\n".join(lines)


# ── AIが出力したJSONを解析 ───────────────────────────────────────────────────

def parse_relationship_updates_from_novel(novel_body: str) -> list[dict]:
    """
    小説本文末尾の ```json ... ``` ブロックから関係性更新を抽出する。
    """
    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    matches = re.findall(pattern, novel_body)
    updates = []
    for m in matches:
        try:
            data = json.loads(m)
            if "relationship_updates" in data:
                updates.extend(data["relationship_updates"])
        except Exception:
            pass
    return updates


# ── D3.js 可視化用データ構築 ──────────────────────────────────────────────────

def build_d3_graph_data(episode_no: int | None = None) -> dict:
    """
    D3.js フォースグラフ用の nodes / links データを返す。
    novel_relationships エッジに加え、文明レジストリの企業も自動的にノードとして追加する。
    """
    edges = get_current_graph(up_to_episode=episode_no)

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
            nodes.append({"id": nid, "label": nid[:12], "group": "company",
                          "color": "#94a3b8", "size": 14})

    # 文明レジストリから企業ノードを追加（エッジなしでも表示）
    # 審査結果に応じて色分け・Tuneへのエッジ自動生成
    _STATUS_COLOR = {
        "active":    "#22c55e",   # 緑 — 活動中
        "collapsed": "#ef4444",   # 赤 — 滅亡
        "ascended":  "#f59e0b",   # 金 — 昇華
        "dormant":   "#64748b",   # グレー — 休眠
    }
    _STATUS_TO_REL = {
        "active":    ("trust",      +2.0, "審査通過・活動中"),
        "collapsed": ("rival",      -3.0, "審査後に滅亡"),
        "ascended":  ("ally",       +4.0, "昇華・繁栄"),
        "dormant":   ("neutral",     0.0, "休眠中"),
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
            nodes.append({
                "id": cid,
                "label": label,
                "group": "company",
                "color": color,
                "size": 14,
                "industry": industry,
                "status": status,
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
            rel_color = REL_TYPES.get(rt, {}).get("color", "#64748b")
            auto_links.append({
                "source": "Tune",
                "target": cid,
                "rel_type": rt,
                "rel_label": REL_TYPES.get(rt, {}).get("label", rt),
                "strength": st,
                "width": max(1.0, abs(st) * 0.6 + 0.5),
                "color": rel_color,
                "opacity": 0.25,
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
