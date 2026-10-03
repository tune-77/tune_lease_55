#!/usr/bin/env python3
"""正規判断資産（data/canonical_judgment_rules.json）の重複を週1回整理する。

方針（2026-10-02 の人手統合 111→83件 と同じ基準・同じデータ形式）:
- 自動統合は「正規化後に同文」のペアだけ。決定的ルールで、LLM/Jev は使わない。
- それ以外の似たペアは「統合候補」として data/judgment_asset_merge_candidates.json に積み、
  /judgment-review で人が承認したら統合する。条件の数値が違うペアは候補にもしない。
- 統合元は削除せず status=merged / merged_into=代表ID で残す。代表は原文を書き換えず、
  統合元にしかない原文を「（同旨: …）」で連結し、原文・出所・使用実績を merged_from に残す。
- 書き換え前に毎回 data/backups/ へバックアップを取る。

使い方:
  .venv/bin/python scripts/judgment_asset_dedup.py --dry-run
  .venv/bin/python scripts/judgment_asset_dedup.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from silent_failure_log import record_silent_failure  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
CANONICAL_JSON = DATA_DIR / "canonical_judgment_rules.json"
CANDIDATES_JSON = DATA_DIR / "judgment_asset_merge_candidates.json"
BACKUP_DIR = DATA_DIR / "backups"
HISTORY_JSONL = DATA_DIR / "judgment_asset_dedup_history.jsonl"
LATEST_JSON = DATA_DIR / "judgment_asset_dedup_latest.json"
CANDIDATE_STATE_JSON = DATA_DIR / "autoresearch_judgment_asset_candidate_state.json"
CANDIDATES_JSONL = DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl"
USAGE_JSONL = DATA_DIR / "judgment_asset_usage_feedback.jsonl"
VAULT_SUBDIR = Path("Projects") / "tune_lease_55" / "Judgment Assets"

# 候補の前段: 各資産の近傍上位K件（埋め込み・文字一致それぞれ）。2026-10-02 の統合ペアで
# 固定閾値（文字0.30/埋込0.75）は 4/39 しか拾えず、近傍K=3 は 35/43 を拾えた。
NEIGHBORS = 3
# Jev が使えない週は、この強い類似だけを候補にする（人が見る件数を抑える）。
JACCARD_MIN = 0.30
EMBEDDING_MIN = 0.75

# ユーザー発言に付く指示・相槌。同文判定のときだけ取り除く（保存する原文は変えない）。
_FILLERS = (
    "判断資産にしておいて",
    "判断資産に入れて",
    "判断資産として覚えといて",
    "覚えておいて",
    "覚えといて",
    "そうだね",
)
_STRIP_RE = re.compile(r"[\s、。，．,.・:：;；!！?？「」『』（）()【】\[\]*＊/／~〜ー-]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or ""))


def normalize_statement(text: str) -> str:
    value = _nfkc(text)
    for filler in _FILLERS:
        value = value.replace(filler, "")
    return _STRIP_RE.sub("", value)


def numbers_in(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(_NUMBER_RE.findall(_nfkc(text)))))


def bigram_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        s = normalize_statement(s)
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(1, len(ga | gb))


def embedding_similarity(texts: list[str]) -> list[list[float]] | None:
    """ローカルの多言語埋め込み（RAGと同じモデル）で類似度行列を返す。使えなければ None。"""
    try:
        from api.knowledge.vector_store import _MODEL_NAME
        from sentence_transformers import SentenceTransformer

        vectors = SentenceTransformer(_MODEL_NAME).encode(texts, normalize_embeddings=True)
        return (vectors @ vectors.T).tolist()
    except Exception as exc:  # noqa: BLE001 - 埋め込みが無くても文字一致だけで動かす
        print(f"[dedup] embedding unavailable: {type(exc).__name__}", file=sys.stderr)
        return None


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _active(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rules if r.get("status") == "active" and r.get("private") is not True]


def pair_id(a: str, b: str) -> str:
    return hashlib.sha256("|".join(sorted((a, b))).encode("utf-8")).hexdigest()[:16]


def _choose_representative(a: dict[str, Any], b: dict[str, Any], *, prefer_short: bool) -> tuple[dict, dict]:
    """同文なら飾りの少ない短い方、似ているだけなら情報の多い長い方を代表にする。"""

    def key(rule: dict[str, Any]) -> tuple:
        length = len(str(rule.get("canonical_statement") or ""))
        return (int(rule.get("user_evidence_count") or 0), int(rule.get("evidence_count") or 0), -length if prefer_short else length)

    return (a, b) if key(a) >= key(b) else (b, a)


# --- 検出 -------------------------------------------------------------------


def detect(
    rules: list[dict[str, Any]],
    *,
    similarity: list[list[float]] | None = None,
) -> dict[str, Any]:
    """active 資産から 自動統合クラスタ と 統合候補ペア を返す。"""
    active = _active(rules)
    norms = [normalize_statement(r.get("canonical_statement") or "") for r in active]
    parent = list(range(len(active)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    texts = [str(r.get("canonical_statement") or "") for r in active]
    jaccard = [[bigram_jaccard(texts[i], texts[j]) if i != j else 0.0 for j in range(len(texts))] for i in range(len(texts))]
    neighbors: set[tuple[int, int]] = set()
    for i in range(len(texts)):
        for matrix in (jaccard, similarity):
            if matrix is None:
                continue
            for j in sorted((j for j in range(len(texts)) if j != i), key=lambda j: -matrix[i][j])[:NEIGHBORS]:
                neighbors.add((min(i, j), max(i, j)))

    candidates: list[dict[str, Any]] = []
    for i, j in combinations(range(len(active)), 2):
        a, b = active[i], active[j]
        ta, tb = texts[i], texts[j]
        if numbers_in(ta) != numbers_in(tb):
            continue  # 数値・条件が違うものは統合しない
        if norms[i] and len(norms[i]) >= 8 and norms[i] == norms[j]:
            parent[find(j)] = find(i)
            continue
        jac = jaccard[i][j]
        emb = similarity[i][j] if similarity is not None else None
        strong = jac >= JACCARD_MIN or (emb is not None and emb >= EMBEDDING_MIN)
        if strong or (i, j) in neighbors:
            rep, src = _choose_representative(a, b, prefer_short=False)
            candidates.append(
                {
                    "id": pair_id(rep["id"], src["id"]),
                    "representative_id": rep["id"],
                    "source_id": src["id"],
                    "representative_statement": rep.get("canonical_statement"),
                    "source_statement": src.get("canonical_statement"),
                    "jaccard": round(jac, 3),
                    "embedding": round(float(emb), 3) if emb is not None else None,
                    "strong_similarity": strong,
                }
            )

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, rule in enumerate(active):
        groups.setdefault(find(i), []).append(rule)
    auto_clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = members[0]
        for other in members[1:]:
            rep, _ = _choose_representative(rep, other, prefer_short=True)
        auto_clusters.append({"representative_id": rep["id"], "source_ids": [m["id"] for m in members if m is not rep]})
    merged_ids = {sid for c in auto_clusters for sid in c["source_ids"]}
    candidates = [c for c in candidates if c["representative_id"] not in merged_ids and c["source_id"] not in merged_ids]
    return {"auto_clusters": auto_clusters, "candidates": candidates}


# --- Jev（候補の並び順・確信度表示だけに使う。自動統合はしない） ---------------------


def build_jev_pair_request(pairs: list[tuple[str, str]], *, model: str = "jev-latest") -> dict[str, Any]:
    questions = {}
    for n in range(len(pairs)):
        questions[f"pair{n}_same_asset"] = {
            "type": "noul",
            "instructions": f"`pairs[{n}].a` と `pairs[{n}].b` は、同じ判断資産として1件に統合してよいか？",
            "criteria": {
                "true": "同じ審査上の判断・注意点を述べており、片方を残せばもう片方は不要。言い回しや補足の違いだけ。",
                "false": "対象（業種・物件・条件）、数値、推奨する行動、または主張の向きが違い、両方残す必要がある。",
            },
        }
    return {"state": {"pairs": [{"a": a[:400], "b": b[:400]} for a, b in pairs]}, "model": model, "questions": questions}


def parse_jev_answers(body: dict[str, Any], count: int) -> list[float]:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("TypeSafe response is missing answers")
    out = []
    for n in range(count):
        raw = answers.get(f"pair{n}_same_asset")
        if not isinstance(raw, dict) or raw.get("type") != "noul":
            raise ValueError(f"missing answer pair{n}_same_asset")
        value = float(raw["noul"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("noul outside [0, 1]")
        out.append(value)
    return out


# --- 出所・使用実績 ------------------------------------------------------------------


class Provenance:
    """統合元の出所（ユーザー発チャット/Auto Research/会話ログ集約）と使用実績を引く。"""

    def __init__(self, state_path: Path = CANDIDATE_STATE_JSON, candidates_path: Path = CANDIDATES_JSONL, usage_path: Path = USAGE_JSONL):
        self.state = _read_json(state_path, {})
        self.candidates: dict[str, dict[str, Any]] = {}
        try:
            for line in candidates_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.candidates[str(row.get("id") or "")] = row
        except OSError:
            pass
        self.by_rule: dict[str, list[str]] = {}
        for key, value in self.state.items():
            if isinstance(value, dict) and value.get("promoted_rule_id"):
                self.by_rule.setdefault(value["promoted_rule_id"], []).append(key)
        self.usage: Counter = Counter()
        self.outcomes: dict[str, Counter] = {}
        try:
            for line in usage_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(row.get("rule_id") or "").removeprefix("cr-")
                self.usage[rid] += 1
                self.outcomes.setdefault(rid, Counter())[str(row.get("outcome") or "")] += 1
        except OSError:
            pass

    def of(self, rule: dict[str, Any]) -> dict[str, Any]:
        rid = str(rule.get("id") or "")
        cids = self.by_rule.get(rid, [])
        cand = next((self.candidates[c] for c in cids if c in self.candidates), {})
        paths = [str(p) for p in rule.get("evidence_paths") or []]
        ar_dates = sorted({m.group(1) for p in paths for m in [re.search(r"Auto Research/(\d{4}-\d{2}-\d{2})", p)] if m})
        chat_dates = sorted({m.group(1) for p in paths for m in [re.search(r"Conversation Log/(\d{4}-\d{2}-\d{2})", p)] if m})
        if cand.get("source_section") == "manual_input" or rule.get("concept") == "chat_judgment_teaching":
            origin, dates = "user_chat", [cand["research_date"]] if cand.get("research_date") else []
        elif ar_dates:
            origin, dates = "auto_research", ar_dates
        else:
            origin, dates = "conversation_log_preview", chat_dates
        st = self.state.get("cr-" + rid, {}) if isinstance(self.state.get("cr-" + rid), dict) else {}
        cst = self.state.get(cids[0], {}) if cids else {}
        return {
            "origin": origin,
            "origin_dates": dates,
            "candidate_ids": cids,
            "usage": {
                "feedback_events": self.usage.get(rid, 0),
                "outcomes": dict(self.outcomes.get(rid, {})),
                "use_count": int(st.get("use_count") or 0) + int(cst.get("use_count") or 0),
                "useful_count": int(st.get("useful_count") or 0) + int(cst.get("useful_count") or 0),
                "rejected_count": int(st.get("rejected_count") or 0) + int(cst.get("rejected_count") or 0),
            },
        }


# --- 統合 -------------------------------------------------------------------------


def _union(a: list | None, b: list | None, limit: int) -> list:
    out = list(a or [])
    for item in b or []:
        if item not in out:
            out.append(item)
    return out[:limit]


def merge_into(
    rules: list[dict[str, Any]],
    representative_id: str,
    source_ids: list[str],
    *,
    reason: str,
    provenance: Provenance | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """統合元を代表へ寄せる（data は呼び出し側で保存する）。統合内容の要約を返す。"""
    now = now or _now()
    by_id = {str(r.get("id") or ""): r for r in rules}
    rep = by_id.get(representative_id)
    sources = [by_id.get(s) for s in source_ids]
    if rep is None or rep.get("status") != "active":
        raise ValueError(f"representative is not active: {representative_id}")
    if any(s is None or s.get("status") != "active" for s in sources) or representative_id in source_ids:
        raise ValueError("every source must be an active rule other than the representative")
    prov = provenance or Provenance(Path("/nonexistent"), Path("/nonexistent"), Path("/nonexistent"))

    old = str(rep.get("canonical_statement") or "")
    extra = [
        str(s.get("canonical_statement") or "")
        for s in sources
        if normalize_statement(s.get("canonical_statement") or "") not in normalize_statement(old)
    ]
    statement = old + "（同旨: " + "／".join(extra) + "）" if extra else old

    merged_from = list(rep.get("merged_from") or [])
    for src in sources:
        merged_from.append(
            {
                "id": src["id"],
                "canonical_statement": src.get("canonical_statement"),
                "concept": src.get("concept"),
                "material_type": src.get("material_type"),
                "evidence_count": src.get("evidence_count", 0),
                "user_evidence_count": src.get("user_evidence_count", 0),
                "created_at": src.get("created_at"),
                "evidence_paths": src.get("evidence_paths", []),
                "merge_reason": reason,
                **prov.of(src),
            }
        )
        rep["evidence_count"] = int(rep.get("evidence_count") or 0) + int(src.get("evidence_count") or 0)
        rep["user_evidence_count"] = int(rep.get("user_evidence_count") or 0) + int(src.get("user_evidence_count") or 0)
        rep["confidence"] = max(float(rep.get("confidence") or 0), float(src.get("confidence") or 0))
        rep["material_types"] = _union(rep.get("material_types"), src.get("material_types") or [src.get("material_type")], 10)
        rep["risk_axis"] = _union(rep.get("risk_axis"), src.get("risk_axis"), 5)
        rep["sample_claims"] = _union(rep.get("sample_claims"), src.get("sample_claims"), 8)
        rep["evidence_paths"] = _union(rep.get("evidence_paths"), src.get("evidence_paths"), 12)
        src.update({"status": "merged", "merged_into": rep["id"], "merged_at": now, "merge_reason": reason, "updated_at": now})
    if statement != old:
        rep.setdefault("pre_merge_statement", old)
        rep["canonical_statement"] = statement
    rep.setdefault("representative_provenance", prov.of(rep))
    rep.update({"merged_from": merged_from, "merged_at": now, "updated_at": now})
    return {"representative_id": rep["id"], "source_ids": list(source_ids), "statement": statement, "reason": reason}


def refresh_summary(store: dict[str, Any], now: str | None = None) -> None:
    rules = store.get("rules") or []
    store["generated_at"] = now or _now()
    store["summary"] = {
        **(store.get("summary") or {}),
        "active_rules": sum(1 for r in rules if r.get("status") == "active"),
        "total_rules": len(rules),
        "merged_rules": sum(1 for r in rules if r.get("status") == "merged"),
    }


def backup(path: Path, label: str, backup_dir: Path = BACKUP_DIR) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{path.stem}.before_{label}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    shutil.copy2(path, target)
    return target


def merge_candidate(
    candidate_id: str,
    *,
    canonical_path: Path = CANONICAL_JSON,
    candidates_path: Path = CANDIDATES_JSON,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    """/judgment-review で人が承認した統合候補を統合する。"""
    queue = _read_json(candidates_path, {"candidates": {}})
    candidate = (queue.get("candidates") or {}).get(candidate_id)
    if not candidate or candidate.get("status") != "pending":
        raise KeyError(candidate_id)
    store = _read_json(canonical_path, {})
    backup_path = backup(canonical_path, "merge_candidate", canonical_path.parent / "backups")
    result = merge_into(
        store.get("rules") or [],
        candidate["representative_id"],
        [candidate["source_id"]],
        reason=f"human_approved_merge_candidate:{candidate_id}",
        provenance=provenance or Provenance(),
    )
    refresh_summary(store)
    _write_json(canonical_path, store)
    candidate.update({"status": "merged", "reviewed_at": _now(), "backup": str(backup_path.relative_to(PROJECT_ROOT)) if backup_path.is_relative_to(PROJECT_ROOT) else str(backup_path)})
    _write_json(candidates_path, queue)
    return {**result, "active_rules": store["summary"]["active_rules"]}


def reject_candidate(candidate_id: str, *, candidates_path: Path = CANDIDATES_JSON, comment: str = "") -> dict[str, Any]:
    queue = _read_json(candidates_path, {"candidates": {}})
    candidate = (queue.get("candidates") or {}).get(candidate_id)
    if not candidate or candidate.get("status") != "pending":
        raise KeyError(candidate_id)
    candidate.update({"status": "rejected", "reviewed_at": _now(), "review_comment": comment[:300]})
    _write_json(candidates_path, queue)
    return candidate


def pending_candidates(
    *, canonical_path: Path = CANONICAL_JSON, candidates_path: Path = CANDIDATES_JSON
) -> list[dict[str, Any]]:
    """両方がまだ active の保留候補だけを、Jev確信度→類似度の順で返す。"""
    queue = _read_json(candidates_path, {"candidates": {}})
    rules = {str(r.get("id") or ""): r for r in _read_json(canonical_path, {}).get("rules") or []}
    out = []
    for candidate in (queue.get("candidates") or {}).values():
        if candidate.get("status") != "pending":
            continue
        rep, src = rules.get(candidate.get("representative_id")), rules.get(candidate.get("source_id"))
        if not rep or not src or rep.get("status") != "active" or src.get("status") != "active":
            continue
        out.append({**candidate, "representative_statement": rep.get("canonical_statement"), "source_statement": src.get("canonical_statement")})
    return sorted(out, key=lambda c: (-(c.get("jev_same_asset") if c.get("jev_same_asset") is not None else -1), -float(c.get("embedding") or 0), -float(c.get("jaccard") or 0)))


# --- 週次実行 ----------------------------------------------------------------------

JevFn = Callable[[list[tuple[str, str]]], list[float]]


def _default_jev(pairs: list[tuple[str, str]]) -> list[float]:
    import typesafe_dedup_guard as transport

    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    scores: list[float] = []
    for start in range(0, len(pairs), 15):
        chunk = pairs[start : start + 15]
        scores += parse_jev_answers(dict(transport._default_request(build_jev_pair_request(chunk))), len(chunk))
    return scores


def _log_jev(candidates: list[dict[str, Any]], *, mode: str, threshold: float) -> None:
    try:
        import jev_judgment_log

        items = [
            {
                "subject": f"{c['representative_statement']}\n{c['source_statement']}",
                "question": "same_judgment_asset",
                "probability": c.get("jev_same_asset"),
                "choice": c.get("jev_same_asset", 0) >= threshold,
                "route": "merge_candidate" if c.get("jev_same_asset", 0) >= threshold else "dropped",
                "auto_passed": False,
                "thresholds": {"drop_below": threshold},
            }
            for c in candidates
            if c.get("jev_same_asset") is not None
        ]
        jev_judgment_log.append_records(
            jev_judgment_log.build_records(guard="judgment_asset_dedup", run_id=jev_judgment_log.new_run_id(), mode=mode, model="jev-latest", items=items)
        )
    except Exception as exc:  # noqa: BLE001 - ログ失敗で整理を止めない
        print(f"[dedup] jev judgment log skipped: {type(exc).__name__}", file=sys.stderr)


def score_candidates_with_jev(candidates: list[dict[str, Any]], jev_fn: JevFn | None, *, drop_below: float | None) -> dict[str, Any]:
    """Jev で候補の確信度を付ける。失敗時は候補をそのまま残す（fail-open）。"""
    if not candidates or jev_fn is None:
        return {"status": "skipped"}
    try:
        import typesafe_dedup_guard as transport

        safe = [c for c in candidates if transport.is_safe_public_candidate({"title": c["representative_statement"]}) and transport.is_safe_public_candidate({"title": c["source_statement"]})]
        scores = jev_fn([(c["representative_statement"], c["source_statement"]) for c in safe])
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": type(exc).__name__}
    for c, score in zip(safe, scores):
        c["jev_same_asset"] = round(score, 3)
    return {"status": "applied", "judged": len(safe), "privacy_skipped": len(candidates) - len(safe), "drop_below": drop_below}


def _vault_dir() -> Path | None:
    try:
        from runtime_paths import resolve_obsidian_vault

        vault = resolve_obsidian_vault()
    except Exception:  # noqa: BLE001 - Vault が無い環境では data/ だけに残す
        return None
    return vault / VAULT_SUBDIR if vault.exists() else None


def _obsidian_note(report: dict[str, Any]) -> str:
    lines = [
        "---",
        f"created: {report['date']}",
        "source: judgment_asset_dedup_weekly",
        "project: tune_lease_55",
        "purpose: judgment_asset_merge_record",
        "rag_connection: none",
        "---",
        "",
        f"# 判断資産 重複整理 週次 {report['date']}",
        "",
        f"- 正規判断資産(active): **{report['before_active']} → {report['after_active']}件**",
        f"- 自動統合（正規化後に同文）: {report['auto_merged_sources']}件 / 統合候補（/judgment-review で承認待ち）: 新規{report['new_candidates']}件・保留合計{report['pending_candidates']}件",
        f"- Jev: {report['jev'].get('status')}（候補の並び順と確信度表示だけに使用。自動統合には使わない）",
        f"- バックアップ: `{report.get('backup') or 'なし（変更なし）'}`",
        "",
        "## 自動統合",
        "",
    ]
    lines += [f"- `{c['representative_id']}` ← {', '.join(f'`{s}`' for s in c['source_ids'])}: {c['statement']}" for c in report["auto_merges"]] or ["- なし"]
    lines += ["", "## 新しい統合候補", ""]
    lines += [
        f"- `{c['id']}` Jev={c.get('jev_same_asset', '-')} 埋込={c.get('embedding', '-')} 文字={c['jaccard']}: {c['representative_statement']} ／ {c['source_statement']}"
        for c in report["new_candidate_list"]
    ] or ["- なし"]
    lines += ["", "## active件数の推移", "", *[f"- {h['date']}: {h['after_active']}件" for h in report["history"][-8:]]]
    return "\n".join(lines) + "\n"


def run(
    *,
    dry_run: bool,
    canonical_path: Path = CANONICAL_JSON,
    candidates_path: Path = CANDIDATES_JSON,
    history_path: Path = HISTORY_JSONL,
    latest_path: Path = LATEST_JSON,
    vault_dir: Path | None = None,
    similarity_fn: Callable[[list[str]], list[list[float]] | None] = embedding_similarity,
    jev_fn: JevFn | None = None,
    jev_drop_below: float | None = None,
    provenance: Provenance | None = None,
    rebuild_index: bool = False,
    today: dt.date | None = None,
) -> dict[str, Any]:
    today = today or dt.date.today()
    store = _read_json(canonical_path, {})
    rules = store.get("rules") or []
    before_active = len(_active(rules))
    active = _active(rules)
    found = detect(rules, similarity=similarity_fn([str(r.get("canonical_statement") or "") for r in active]))

    queue = _read_json(candidates_path, {"candidates": {}})
    known = queue.setdefault("candidates", {})
    new_candidates = [c for c in found["candidates"] if c["id"] not in known]
    jev = score_candidates_with_jev(new_candidates, jev_fn, drop_below=jev_drop_below)
    dropped: list[dict[str, Any]] = []
    if jev.get("status") == "applied":
        if jev_drop_below is not None:
            dropped = [c for c in new_candidates if c.get("jev_same_asset") is not None and c["jev_same_asset"] < jev_drop_below]
            new_candidates = [c for c in new_candidates if c not in dropped]
        # Jev 判定できなかった（プライバシー除外）ペアは強い類似のときだけ残す。
        new_candidates = [c for c in new_candidates if c.get("jev_same_asset") is not None or c["strong_similarity"]]
    else:
        # Jev なしの週は強い類似だけ。近傍だけのペアは保存せず、次回 Jev で判定する。
        new_candidates = [c for c in new_candidates if c["strong_similarity"]]
    jev["dropped"] = len(dropped)

    auto_merges: list[dict[str, Any]] = []
    backup_path = None
    if not dry_run:
        if found["auto_clusters"]:
            backup_path = backup(canonical_path, "weekly_dedup", canonical_path.parent / "backups")
            prov = provenance or Provenance()
            for cluster in found["auto_clusters"]:
                auto_merges.append(merge_into(rules, cluster["representative_id"], cluster["source_ids"], reason="weekly_dedup_identical_text", provenance=prov))
            refresh_summary(store)
            _write_json(canonical_path, store)
        now = _now()
        for c in new_candidates:
            known[c["id"]] = {**c, "status": "pending", "created_at": now}
        for c in dropped:  # 再判定しないよう記録だけ残す（画面には出さない）
            known[c["id"]] = {**c, "status": "dropped_by_jev", "created_at": now}
        _write_json(candidates_path, queue)
        if jev.get("status") == "applied":
            _log_jev(new_candidates + dropped, mode="weekly", threshold=jev_drop_below or 0.0)
    else:
        auto_merges = [{**c, "statement": "(dry-run)"} for c in found["auto_clusters"]]

    after_active = len(_active(rules))
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()] if history_path.exists() else []
    pending = len([c for c in known.values() if c.get("status") == "pending"]) if not dry_run else None
    entry = {
        "date": today.isoformat(),
        "before_active": before_active,
        "after_active": after_active,
        "auto_merged_sources": sum(len(c["source_ids"]) for c in auto_merges),
        "new_candidates": len(new_candidates),
        "pending_candidates": pending,
    }
    report = {
        **entry,
        "dry_run": dry_run,
        "generated_at": _now(),
        "backup": str(backup_path.relative_to(PROJECT_ROOT)) if backup_path and backup_path.is_relative_to(PROJECT_ROOT) else (str(backup_path) if backup_path else None),
        "jev": jev,
        "auto_merges": auto_merges,
        "new_candidate_list": new_candidates,
        "history": history + [entry],
    }
    if dry_run:
        return report

    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _write_json(latest_path.parent / f"judgment_asset_dedup_weekly_{today.strftime('%Y%m%d')}.json", report)
    _write_json(latest_path, report)
    vault = vault_dir if vault_dir is not None else _vault_dir()
    if vault is not None:
        vault.mkdir(parents=True, exist_ok=True)
        (vault / f"判断資産 重複整理 週次 {today.isoformat()}.md").write_text(_obsidian_note(report), encoding="utf-8")
    if rebuild_index and auto_merges:
        # 想起で統合元を引かないよう、記憶索引を作り直す（日次パイプラインと同じスクリプト）。
        rebuilt = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_shion_memory_index.py")], check=False)
        if rebuilt.returncode != 0:  # 作り直せないと統合元の記憶を想起し続ける
            record_silent_failure("memory.judgment_asset_dedup.rebuild_index", "subprocess_failed", detail=f"exit {rebuilt.returncode}")
    return report


def morning_report_line(latest_path: Path = LATEST_JSON) -> str:
    """AURION CORE 朝報向けの1行。"""
    report = _read_json(latest_path, None)
    if not isinstance(report, dict):
        return "- 判断資産の重複整理（週次）: まだ実行されていません"
    return (
        f"- 判断資産の重複整理（週次 {report.get('date')}）: active {report.get('before_active')}→{report.get('after_active')}件、"
        f"自動統合 {report.get('auto_merged_sources')}件、統合候補 {report.get('pending_candidates')}件（/judgment-review）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="判断資産の重複を週次で整理する")
    parser.add_argument("--dry-run", action="store_true", help="検出だけして何も書き換えない")
    parser.add_argument("--no-jev", action="store_true", help="Jev で候補の確信度を付けない")
    args = parser.parse_args()
    import typesafe_dedup_guard as transport

    use_jev = not args.no_jev and bool(transport._resolve_api_key())
    report = run(
        dry_run=args.dry_run,
        jev_fn=_default_jev if use_jev else None,
        jev_drop_below=JEV_DROP_BELOW,
        rebuild_index=True,
    )
    summary = {k: report[k] for k in ("date", "dry_run", "before_active", "after_active", "auto_merged_sources", "new_candidates", "pending_candidates", "backup")}
    print(json.dumps({**summary, "jev": report["jev"], "auto_merges": report["auto_merges"]}, ensure_ascii=False, indent=2))
    # 同じ週次ジョブで、紫苑のユーザー個人記憶も同じ考え方（削除せずアーカイブ・迷うものは候補）で整理する
    try:
        from api.user_personal_memory_archive import run as run_personal_memory_hygiene

        personal = run_personal_memory_hygiene(
            dry_run=args.dry_run,
            similarity_fn=embedding_similarity,
            pair_scorer=(lambda texts, question: transport.judge_binary_pairs(texts, question)[0]) if use_jev else None,
        )
        print(json.dumps({"user_personal_memory": {k: v for k, v in personal.items() if k != "auto"}}, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 - 個人記憶の整理の失敗で判断資産の整理結果を落とさない
        print(f"[user_personal_memory] 整理失敗: {type(exc).__name__}: {exc}", file=sys.stderr)


# 実データ評価（experiments/judgment_asset_dedup_jev/README.md）: 0.2 未満で落としても
# 2026-10-02 の統合ペア 39/39 は残り、統合しなかった近いペアの 21/55 を落とせた。
JEV_DROP_BELOW: float | None = 0.2


if __name__ == "__main__":
    main()
