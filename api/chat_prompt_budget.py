"""チャット・対話室のシステムプロンプトを、ブロックごとの優先度と文字数の予算で組み立てる。

旧方式（api/main.py の _cap_system_prompt）は上限24,000字を超えると末尾から段落を削っていた。
2026-10-03 の計測（前後比較の10問＋雑談3問）で、通常チャットは毎回 2,700〜11,900字が削られ、
RAG/Knowledge の95%・回答の形の指示の93%・記憶の想起の62%が消えていた。一方で
ユーザー個人記憶は毎回 9,130字（予算の38%）を占めていた。対話室には上限が無く平均33,600字だった。

方式:
1. 重複の除去: 同じ項目（判断資産など）が複数のブロックにあれば、優先度の高いブロックの1つだけ残す
2. ブロックごとの予算: 記憶の想起〜補助の層は常に、方針〜根拠の層は合計が上限を超えた時だけ、
   予算を超えたブロックを項目（段落・箇条書き）単位で関連度の低いものから落とす
   - ordered: 既に関連度順に並んでいる（RAG・想起）→ 末尾から落とす
   - relevance: 質問との文字bigram一致率が低い項目から落とす（残す項目の元の順番は保つ）
   - oldest_first: 会話履歴 → 古い（先頭の）項目から落とす
3. それでも合計が上限を超えたら、優先度の低い層から順に削る（core は削らない）
並び順は変えない（人格・口調への影響を避けるため。削られなくなったので順番は重み付けだけに効く）。
社内方針の節は reserved_tail として必ず末尾に付ける（PR #1225）。
落とした量はブロック名と文字数だけを data/chat_prompt_budget_log.jsonl に記録する（本文は残さない）。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 層: 0=core（削らない） 1=社内方針・教えたノウハウ・判断資産 2=回答の形 3=RAG/Knowledge・Vertex
#     4=記憶の想起 5=会話履歴 6=経験ループ・ニュース等の補助
CORE, TAUGHT, SHAPE, EVIDENCE, MEMORY, HISTORY, AUX = range(7)


@dataclass(frozen=True)
class BlockSpec:
    tier: int
    budget: int | None = None  # None は予算なし（core・小さいブロック）
    mode: str = "relevance"  # ordered / relevance / oldest_first


SPECS: dict[str, BlockSpec] = {
    # core: 人格・システム指示・モード・紫苑の同一性
    "base_prompt_root": BlockSpec(CORE),
    "base_system_root": BlockSpec(CORE),
    "dialogue_base": BlockSpec(CORE),  # 旧呼び出し互換
    "dialogue_identity_core": BlockSpec(CORE),
    "dialogue_self_context": BlockSpec(MEMORY, 3500),
    "dialogue_execution_core": BlockSpec(CORE),
    "dialogue_knowledge_context": BlockSpec(EVIDENCE, 7000, "ordered"),
    "dialogue_response_core": BlockSpec(CORE),
    "mode_instruction": BlockSpec(CORE),
    "response_mode_context": BlockSpec(CORE),
    "identity_memory_context": BlockSpec(CORE),
    "teaching_save_context": BlockSpec(CORE),
    # 1: ユーザーが教えたノウハウ・判断資産・基本知識
    "teaching_prompt_context": BlockSpec(TAUGHT, 2500, "ordered"),
    "pre_recall_context": BlockSpec(TAUGHT, 2500, "ordered"),
    "basic_lease_question_prompt": BlockSpec(TAUGHT, 2000),
    "judgment_learning_context": BlockSpec(TAUGHT, 1500, "ordered"),
    "grey_judgment_context": BlockSpec(TAUGHT, 1500, "ordered"),
    "business_plan_consult_context": BlockSpec(TAUGHT, 1500),
    # 2: 回答の形の指示
    "judgment_response_shape_context": BlockSpec(SHAPE, 1200),
    "case_screening_pattern_context": BlockSpec(SHAPE, 1200),
    "case_screening_mentor_dialogue_context": BlockSpec(SHAPE, 1000),
    "shion_specificity_context": BlockSpec(SHAPE, 800),
    "vague_information_request_context": BlockSpec(SHAPE, 600),
    "shion_light_tone_context": BlockSpec(SHAPE, 600),
    "shion_non_domain_context": BlockSpec(SHAPE, 600),
    "prompt_suffix": BlockSpec(SHAPE, 600),
    # 3: RAG/Knowledge・Vertex の根拠・DB・外部調査
    "rag_context": BlockSpec(EVIDENCE, 5000, "ordered"),
    "db_context": BlockSpec(EVIDENCE, 1200, "ordered"),
    "external_research_context": BlockSpec(EVIDENCE, 1500, "ordered"),
    "news_focus_context": BlockSpec(EVIDENCE, 800, "ordered"),
    # 4: 記憶の想起
    "memory_recall_context": BlockSpec(MEMORY, 2200, "ordered"),
    "shared_shion_memory_context": BlockSpec(MEMORY, 6000),
    "mid_term_memory_context": BlockSpec(MEMORY, 600, "ordered"),
    "memory_to_judgment_context": BlockSpec(MEMORY, 400),
    "memory_expression_context": BlockSpec(MEMORY, 600),
    "improvement_context": BlockSpec(MEMORY, 1500, "ordered"),
    "pdca_block": BlockSpec(MEMORY, 800),
    # 5: 会話履歴の要約
    "chat_history_summary_context": BlockSpec(HISTORY, 1500, "oldest_first"),
    # 6: 補助
    "user_personal_memory_context": BlockSpec(AUX, 3500),
    "news_brief_context": BlockSpec(AUX, 600, "ordered"),
    "news_actions_context": BlockSpec(AUX, 500, "ordered"),
    "news_digest_context": BlockSpec(AUX, 500, "ordered"),
    "obsidian_daily_context": BlockSpec(AUX, 800, "ordered"),
    "experience_loop_context": BlockSpec(AUX, 600),
    "continuity_hook_context": BlockSpec(AUX, 400),
    "delta_awareness_context": BlockSpec(AUX, 500),
    "reflection_gate_context": BlockSpec(AUX, 500),
    "world_proxy_context": BlockSpec(AUX, 600),
    "consciousness_ux_context": BlockSpec(AUX, 900),
    "human_device_resonance_context": BlockSpec(AUX, 400),
    "improvement_report_context": BlockSpec(AUX, 800, "ordered"),
    "improvement_observability_context": BlockSpec(AUX, 600, "ordered"),
    "agent_consultation_context": BlockSpec(AUX, 800, "ordered"),
    "reasoner_consultation_context": BlockSpec(AUX, 800, "ordered"),
    "improvement_triage_context": BlockSpec(AUX, 500, "ordered"),
}
DEFAULT_SPEC = BlockSpec(AUX, 800)

_LOG_PATH = Path(os.environ.get("CHAT_PROMPT_BUDGET_LOG_PATH") or Path(__file__).resolve().parents[1] / "data" / "chat_prompt_budget_log.jsonl")
_HEADER_RE = re.compile(r"^\s*【[^】]{1,60}】")
_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-*・]|\d+[.)]|\[[^\]]{1,40}\])\s*")
_CITE_RE = re.compile(r"（出典[:：][^）]*）")


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s、。,.()（）「」『』:：/・*#>\-`【】\[\]]", "", str(text or ""))
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _split_items(text: str) -> tuple[str, list[str]]:
    """先頭の【見出し】行（と直後の説明行）を固定部分、残りを項目（段落、なければ行）に分ける。"""
    body = str(text or "")
    lead = ""
    stripped = body.lstrip("\n")
    lead_ws = body[: len(body) - len(stripped)]
    if _HEADER_RE.match(stripped):
        first_nl = stripped.find("\n")
        lead = stripped if first_nl < 0 else stripped[:first_nl]
        stripped = "" if first_nl < 0 else stripped[first_nl + 1 :]
    paragraphs = [p for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    items = paragraphs if len(paragraphs) > 1 else [line for line in stripped.split("\n") if line.strip()]
    return lead_ws + lead, items


def _join(lead: str, items: list[str]) -> str:
    sep = "\n\n" if any("\n" in it for it in items) else "\n"
    body = sep.join(items)
    if lead.strip():
        return f"{lead}\n{body}" if body else lead
    return (lead + body) if body else ""


def _item_key(item: str) -> str:
    text = _CITE_RE.sub("", _ITEM_PREFIX_RE.sub("", item.strip()))
    text = re.sub(r"[\s、。,.（）()「」]", "", text)
    return text[:50]


def _fit(text: str, budget: int, mode: str, question_grams: set[str]) -> str:
    if budget is None or len(text) <= budget:
        return text
    lead, items = _split_items(text)
    if not items:
        return text[:budget]
    if mode == "oldest_first":
        while items and len(_join(lead, items)) > budget:
            items.pop(0)
    elif mode == "ordered":
        while items and len(_join(lead, items)) > budget:
            items.pop()
    else:
        scored = sorted(
            range(len(items)),
            key=lambda i: (len(_bigrams(items[i]) & question_grams) / max(1, len(question_grams)), -i),
        )
        alive = set(range(len(items)))
        for i in scored:
            if len(_join(lead, [items[j] for j in sorted(alive)])) <= budget:
                break
            alive.discard(i)
        items = [items[j] for j in sorted(alive)]
    result = _join(lead, items)
    return result if len(result) <= budget or not items else result[:budget]


def split_dialogue_prompt(prompt: str) -> list[tuple[str, str]]:
    """対話室の一体化したpromptを固定指示と動的な自己状態・知識へ分ける。"""
    body = str(prompt or "")
    markers = ("【自己状態】", "【実行環境】", "【関連するObsidian知識】", "【今回の応答モード】")
    positions = [body.find(marker) for marker in markers]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        return [("dialogue_base", body)]
    self_pos, env_pos, knowledge_pos, response_pos = positions
    intro_end = body.find("\n\n")
    if intro_end < 0 or intro_end > self_pos:
        intro_end = self_pos
    else:
        intro_end += 2
    return [
        ("dialogue_identity_core", body[:intro_end]),
        ("dialogue_self_context", body[intro_end:env_pos]),
        ("dialogue_execution_core", body[env_pos:knowledge_pos]),
        ("dialogue_knowledge_context", body[knowledge_pos:response_pos]),
        ("dialogue_response_core", body[response_pos:]),
    ]


def _fit_reserved_tail(text: str, limit: int) -> str:
    """極端に小さい設定でも全体上限を守り、方針行を説明文より優先する。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    essential = [line for line in lines if line.startswith(("【", "- 方針: "))]
    compact = "\n".join(essential) or text
    return compact[:limit]


def assemble_prompt(
    blocks: list[tuple[str, str]],
    *,
    question: str,
    surface: str,
    max_chars: int | None = None,
    reserved_tail: str = "",
    log: bool = True,
) -> tuple[str, dict[str, Any]]:
    """(ブロック名, 本文) の並びから、予算内のプロンプトと落とした量のレポートを返す。"""
    if max_chars is None:
        env_name = "DIALOGUE_SYSTEM_PROMPT_MAX_CHARS" if surface.startswith("dialogue") else "CHAT_SYSTEM_PROMPT_MAX_CHARS"
        max_chars = int(os.environ.get(env_name, "30000" if surface.startswith("dialogue") else "24000"))
    original_tail = str(reserved_tail or "").strip()
    tail = _fit_reserved_tail(original_tail, max_chars)
    grams = _bigrams(question)
    report: dict[str, dict[str, int]] = {}
    texts: dict[int, str] = {}

    # 1. 初期化。個別予算を適用してから重複除去することで、高優先度側で予算落ちした項目を
    # 低優先度側のコピーからも先に消してしまうことを防ぐ。
    for i, (name, text) in enumerate(blocks):
        text = str(text or "")
        spec = SPECS.get(name, DEFAULT_SPEC)
        report[name] = {"tier": spec.tier, "orig": len(text), "dedup": 0, "budget_cut": 0, "overflow_cut": 0, "kept": 0}
        texts[i] = text
    original_occurrences: dict[str, list[tuple[int, str]]] = {}
    for i, (name, _text) in enumerate(blocks):
        if SPECS.get(name, DEFAULT_SPEC).tier == CORE:
            continue
        _lead, items = _split_items(texts[i])
        for item in items:
            key = _item_key(item)
            if len(key) >= 20:
                original_occurrences.setdefault(key, []).append((i, item))

    limit = max(0, max_chars - (len(tail) + 2 if tail else 0))

    def apply_budgets(tiers: range) -> None:
        for i, (name, _text) in enumerate(blocks):
            spec = SPECS.get(name, DEFAULT_SPEC)
            if spec.tier in tiers and spec.budget and len(texts[i]) > spec.budget:
                before = len(texts[i])
                texts[i] = _fit(texts[i], spec.budget, spec.mode, grams)
                report[name]["budget_cut"] += before - len(texts[i])

    def overflow(tiers: range) -> None:
        total = sum(len(t) for t in texts.values())
        for tier in sorted(tiers, reverse=True):
            for i in sorted((i for i in texts if SPECS.get(blocks[i][0], DEFAULT_SPEC).tier == tier), reverse=True):
                if total <= limit:
                    return
                before = len(texts[i])
                target = max(0, before - (total - limit))
                texts[i] = _fit(texts[i], target, SPECS.get(blocks[i][0], DEFAULT_SPEC).mode, grams) if target else ""
                report[blocks[i][0]]["overflow_cut"] += before - len(texts[i])
                total -= before - len(texts[i])

    def deduplicate() -> None:
        order = sorted(range(len(blocks)), key=lambda i: (SPECS.get(blocks[i][0], DEFAULT_SPEC).tier, i))
        seen: set[str] = set()
        for i in order:
            name, _text = blocks[i]
            spec = SPECS.get(name, DEFAULT_SPEC)
            text = texts[i]
            if not text or spec.tier == CORE:
                continue
            lead, items = _split_items(text)
            kept_items = []
            for item in items:
                key = _item_key(item)
                if len(key) >= 20 and key in seen:
                    continue
                seen.add(key)
                kept_items.append(item)
            deduped = text if len(kept_items) == len(items) else _join(lead, kept_items)
            report[name]["dedup"] += len(text) - len(deduped)
            texts[i] = deduped

    def restore_trimmed_duplicate_fallbacks() -> None:
        """上位コピーが予算落ちした重複項目を、空きがあれば下位コピーから戻す。"""
        live_keys = {
            _item_key(item)
            for i, (name, _text) in enumerate(blocks)
            if SPECS.get(name, DEFAULT_SPEC).tier != CORE
            for item in _split_items(texts[i])[1]
        }
        total = sum(len(t) for t in texts.values())
        for key, occurrences in original_occurrences.items():
            if len(occurrences) < 2 or key in live_keys:
                continue
            i, item = max(occurrences, key=lambda pair: (SPECS.get(blocks[pair[0]][0], DEFAULT_SPEC).tier, pair[0]))
            addition = ("\n" if texts[i] else "") + item
            if total + len(addition) <= limit:
                texts[i] += addition
                total += len(addition)
                live_keys.add(key)

    # 2. 記憶の想起〜補助は常に予算内へ（太りやすい層）。3. 超過分はまず低い層から削る
    apply_budgets(range(MEMORY, AUX + 1))
    overflow(range(MEMORY, AUX + 1))
    # 3. それでも超える時だけ、方針〜根拠の層にも予算を当てる。
    if sum(len(t) for t in texts.values()) > limit:
        apply_budgets(range(TAUGHT, EVIDENCE + 1))
        overflow(range(TAUGHT, EVIDENCE + 1))
    # 4. 高優先度側で予算落ちした重複項目は、低優先度側に収まる時だけ救済してから重複除去する。
    restore_trimmed_duplicate_fallbacks()
    deduplicate()
    # 設定上限が固定指示やtailより小さい場合だけの最後の安全弁。
    if sum(len(t) for t in texts.values()) > limit:
        overflow(range(CORE, CORE + 1))

    prompt = "".join(texts[i] for i in range(len(blocks)))
    if tail:
        prompt = f"{prompt}\n\n{tail}" if prompt else tail
    for i in range(len(blocks)):
        report[blocks[i][0]]["kept"] = len(texts[i])
    summary = {
        "surface": surface,
        "max_chars": max_chars,
        "final_chars": len(prompt),
        "orig_chars": sum(r["orig"] for r in report.values()) + (len(original_tail) + 2 if original_tail else 0),
        "dropped_chars": sum(r["orig"] - r["kept"] for r in report.values()),
        "dropped_high_priority_chars": sum(r["orig"] - r["kept"] - r["dedup"] for r in report.values() if r["tier"] <= EVIDENCE),
        "policy_tail_chars": len(tail),
        "policy_tail_cut": len(original_tail) - len(tail),
        "blocks": {k: v for k, v in report.items() if v["orig"]},
    }
    if log:
        _append_log(summary)
    return prompt, summary


def _append_log(summary: dict[str, Any]) -> None:
    """ブロック名と文字数だけを記録する（本文・質問文は残さない）。"""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **summary}
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def morning_report_line(path: Path | None = None, *, now: dt.datetime | None = None, threshold: int = 4000) -> str | None:
    """直近24時間で落とした量が大きい日だけ、朝報に1行（重要度の高いブロックが削られた回数も出す）。"""
    now = now or dt.datetime.now().astimezone()
    since = now - dt.timedelta(hours=24)
    rows = []
    try:
        for line in (path or _LOG_PATH).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if dt.datetime.fromisoformat(row["ts"]) >= since:
                rows.append(row)
    except (OSError, KeyError, ValueError):
        return None
    if not rows:
        return None
    avg_drop = sum(r.get("dropped_chars", 0) for r in rows) / len(rows)
    high = [r for r in rows if r.get("dropped_high_priority_chars", 0) > 0]
    if avg_drop < threshold and not high:
        return None
    worst: dict[str, int] = {}
    for r in rows:
        for name, b in (r.get("blocks") or {}).items():
            worst[name] = worst.get(name, 0) + b.get("orig", 0) - b.get("kept", 0) - b.get("dedup", 0)
    top = ", ".join(f"{n} {v // len(rows):,}字" for n, v in sorted(worst.items(), key=lambda kv: -kv[1])[:3] if v > 0)
    return (
        f"- ✂️ チャットのプロンプト削減（直近24h・{len(rows)}回）: 平均 {avg_drop:,.0f}字を削除"
        + (f"・重要ブロック（方針〜根拠）が削られた回 {len(high)}回" if high else "")
        + (f"（多い順: {top}）" if top else "")
    )
