"""Small Obsidian bridge for the mobile API chat.

The bridge stores only summarized notes selected by the AI chat layer. It does
not expose raw vault contents wholesale.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from obsidian_query import split_query_terms

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def _obsidian_app_vaults() -> list[Path]:
    """Obsidianアプリの設定ファイルから登録済みVaultパスを返す。
    open=True のVaultを優先し、同条件ならts（最終アクセス時刻）が新しい順。
    """
    import json
    config = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
    if not config.exists():
        return []
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
        vaults = data.get("vaults", {})
        # open=True を優先（1→0）、同条件はts降順
        entries = sorted(
            vaults.values(),
            key=lambda v: (0 if v.get("open") else 1, -v.get("ts", 0)),
        )
        return [Path(v["path"]) for v in entries if v.get("path") and Path(v["path"]).exists()]
    except Exception:
        return []


def _home_candidates() -> list[Path]:
    home = Path.home()
    # 知識宇宙では obsidian-vault を優先し、なければ従来の Obsidian Vault にフォールバックする。
    # lease-wiki-vault はユーザーが明示指定した場合だけ使うため、アプリの最近使用順より後に置く。
    app_vaults = _obsidian_app_vaults()
    env_vault = Path(os.getenv("OBSIDIAN_VAULT", "")).expanduser() if os.getenv("OBSIDIAN_VAULT") else None
    icloud_docs = home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    default_vaults = [
        icloud_docs / "obsidian-vault",
        icloud_docs / "Obsidian Vault",
        home / "Documents" / "Obsidian Vault",
    ]
    regular_app_vaults = [p for p in app_vaults if p.name != "lease-wiki-vault"]
    lease_wiki_vaults = [p for p in app_vaults if p.name == "lease-wiki-vault"]
    roots = [
        env_vault,
        *default_vaults,
        *regular_app_vaults,
        home / "Documents",
        home / "Obsidian",
        icloud_docs,
        home / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
        *lease_wiki_vaults,
    ]
    return [p for p in roots if p and p.exists()]


def find_vault() -> Path | None:
    vaults: list[Path] = []
    for root in _home_candidates():
        if (root / ".obsidian").exists():
            vaults.append(root.resolve())
            continue
        try:
            for marker in root.rglob(".obsidian"):
                if marker.is_dir():
                    vaults.append(marker.parent.resolve())
                    break
        except OSError:
            continue
    seen: set[str] = set()
    for vault in vaults:
        key = str(vault)
        if key not in seen:
            return vault
        seen.add(key)
    return None


def _safe_note_path(vault: Path, rel: str) -> Path:
    target = (vault / rel).resolve()
    if vault not in target.parents and target != vault:
        raise ValueError("refusing to write outside the Obsidian vault")
    if target.suffix.lower() != ".md":
        target = target.with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _to_wikilink(rel_path: str, alias: str | None = None) -> str:
    stem = str(rel_path or "").strip()
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    stem = stem.strip().replace("\\", "/")
    if not stem:
        return ""
    if alias:
        return f"[[{stem}|{alias}]]"
    return f"[[{stem}]]"


_GENERIC_SEARCH_TERMS = {
    "確認", "確認事項", "注意", "注意点", "ポイント", "リスク", "関係", "違い",
    "意味", "見方", "方法", "理由", "原因", "対策", "手順", "選択肢", "営業説明",
    "案件", "会社", "場合", "するとき", "するときの",
}
_CHAT_INTENT_TERMS = ("チャット", "会話", "履歴", "日報", "daily", "weekly review", "改善ログ")
_HUMOR_INTENT_TERMS = ("ユーモア", "笑い", "面白", "口調")
_REFERENCE_PATH_PREFIXES = ("リース知識/", "projects/tune_lease_55/", "projects/tune-lease-55/")
_LOWER_PRIORITY_PATH_PARTS = ("05-クリップ_記事/", "06-日記_作業ログ/", "/news/")
_SOURCE_PRIORITY_RULES = (
    ("リース知識/", 1.00),
    ("projects/tune_lease_55/asset knowledge/", 0.95),
    ("projects/tune_lease_55/asset finance/", 0.92),
    ("projects/tune_lease_55/cases/", 0.90),
    ("projects/tune_lease_55/feedback/", 0.82),
    ("projects/tune_lease_55/research/", 0.76),
    ("projects/tune_lease_55/", 0.70),
    ("07-アーカイブ/asset knowledge/", 0.68),
    ("06-日記_作業ログ/", 0.48),
    ("projects/tune_lease_55/ai chat/", 0.42),
    ("daily/", 0.35),
)
_NOISE_PATH_PARTS = (
    "weekly review",
    "improvement log",
    "検索語インデックス",
    "05-クリップ_記事/",
)


def _normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = re.sub(r"[‐‑‒–—―−_]+", "-", normalized)
    return re.sub(r"\s+", " ", normalized)


def _normalize_path_text(path: str) -> str:
    normalized = unicodedata.normalize("NFKC", path or "").lower().replace("\\", "/")
    normalized = re.sub(r"[‐‑‒–—―−]+", "-", normalized)
    return re.sub(r"\s+", " ", normalized)


def _candidate_score(
    *,
    path: str,
    text: str,
    primary_terms: list[str],
    expanded_terms: list[str],
    query: str,
    source_bonus: float = 0.0,
) -> float:
    normalized_path = _normalize_search_text(path)
    filename = _normalize_search_text(Path(path).stem)
    body = _normalize_search_text(text)
    normalized_query = _normalize_search_text(query)
    primary = [_normalize_search_text(t) for t in primary_terms if t not in _GENERIC_SEARCH_TERMS]
    secondary = [_normalize_search_text(t) for t in expanded_terms if t not in primary_terms]

    score = source_bonus
    matched_primary = 0
    for term in primary:
        if not term:
            continue
        matched = False
        if term in filename:
            score += 18
            matched = True
        elif term in normalized_path:
            score += 10
            matched = True
        if term in body:
            score += min(body.count(term), 3) * 3
            matched = True
        if matched:
            matched_primary += 1

    for term in secondary:
        if not term or term in _GENERIC_SEARCH_TERMS:
            continue
        if term in filename:
            score += 12
        elif term in normalized_path:
            score += 5
        elif term in body:
            score += 1

    if primary:
        coverage = matched_primary / len(primary)
        score += coverage * 16
        if matched_primary == len(primary):
            score += 8
    if len(normalized_query) >= 4 and normalized_query in body:
        score += 5

    is_chat_log = any(f"/{folder.lower()}/" in f"/{normalized_path}/" for folder in _CHAT_LOG_DIRS)
    asks_for_chat = any(term in normalized_query for term in _CHAT_INTENT_TERMS)
    if is_chat_log and not asks_for_chat:
        score -= 24
    elif not is_chat_log:
        score += 5

    if normalized_path.startswith(_REFERENCE_PATH_PREFIXES):
        score += 7
    if any(part in normalized_path for part in _LOWER_PRIORITY_PATH_PARTS):
        score -= 8

    is_humor = "humor/" in normalized_path or "ユーモア" in normalized_path or "八奈見" in path
    asks_for_humor = any(term in normalized_query for term in _HUMOR_INTENT_TERMS)
    if is_humor and not asks_for_humor:
        score -= 60
    elif is_humor and asks_for_humor:
        score += 25

    if "cases/" in normalized_path and any(
        term in normalized_query for term in ("過去", "類似", "スコア", "判定", "金利", "q-risk", "q_risk")
    ):
        score += 12
    if "asset knowledge/" in normalized_path and any(
        term in normalized_query for term in ("物件", "残価", "中古", "売却", "換金", "再販")
    ):
        score += 12
    if "業種別" in normalized_path and any(term in normalized_query for term in ("業", "業種")):
        score += 10
    return score


def _path_source_priority(path: str) -> float:
    normalized_path = _normalize_path_text(path)
    for prefix, priority in _SOURCE_PRIORITY_RULES:
        if normalized_path.startswith(prefix):
            return priority
    return 0.55


def _query_intent_bonus(path: str, query: str) -> float:
    normalized_path = _normalize_path_text(path)
    normalized_query = _normalize_search_text(query)
    bonus = 0.0
    if "cases/" in normalized_path and any(
        term in normalized_query for term in ("過去", "類似", "案件", "事例", "前回", "スコア", "判定", "金利")
    ):
        bonus += 0.22
    if "asset knowledge/" in normalized_path and any(
        term in normalized_query for term in ("物件", "残価", "中古", "売却", "換金", "再販", "処分")
    ):
        bonus += 0.22
    if "research/" in normalized_path and any(
        term in normalized_query for term in ("補助金", "助成金", "税制", "ニュース", "調査", "市場")
    ):
        bonus += 0.12
    if normalized_path.startswith("リース知識/") and any(
        term in normalized_query for term in ("審査", "確認", "条件", "承認", "格付", "q-risk", "q_risk")
    ):
        bonus += 0.16
    return min(bonus, 0.30)


def _noise_penalty(path: str, query: str) -> float:
    normalized_path = _normalize_path_text(path)
    normalized_query = _normalize_search_text(query)
    penalty = 0.0
    asks_for_chat = any(term in normalized_query for term in _CHAT_INTENT_TERMS)
    asks_for_humor = any(term in normalized_query for term in _HUMOR_INTENT_TERMS)
    if not asks_for_chat and any(part in normalized_path for part in ("ai chat/", "daily/", "weekly review", "improvement log")):
        penalty += 0.28
    if not asks_for_humor and any(part in normalized_path for part in ("humor/", "ユーモア", "八奈見")):
        penalty += 0.70
    if any(part in normalized_path for part in _NOISE_PATH_PARTS):
        penalty += 0.12
    if "wiki" in normalized_path and "wiki" not in normalized_query:
        penalty += 0.04
    return min(penalty, 0.90)


def _term_coverage_score(path: str, text: str, primary_terms: list[str]) -> tuple[float, int, int]:
    primary = [_normalize_search_text(t) for t in primary_terms if t and t not in _GENERIC_SEARCH_TERMS]
    if not primary:
        return 0.0, 0, 0
    haystack = f"{_normalize_search_text(path)} {_normalize_search_text(Path(path).stem)} {_normalize_search_text(text)}"
    matched = sum(1 for term in primary if term and term in haystack)
    coverage = matched / len(primary)
    return coverage, matched, len(primary)


def _semantic_score_from_item(item: dict[str, Any] | None, vector_rank: int | None = None) -> float:
    if not item:
        return 0.0
    if item.get("rank_score") is not None:
        try:
            # vector_store rank_score is centered around similarity + business priority.
            return max(0.0, min(1.0, (float(item["rank_score"]) + 1.0) / 2.0))
        except (TypeError, ValueError):
            pass
    if item.get("distance") is not None:
        try:
            return max(0.0, min(1.0, 1.0 - float(item["distance"])))
        except (TypeError, ValueError):
            pass
    if vector_rank is not None:
        return max(0.0, 1.0 - vector_rank * 0.04)
    return 0.0


def _rerank_obsidian_candidates(
    candidates: dict[str, dict[str, Any]],
    *,
    query: str,
    primary_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    raw_scores = [float(item.get("score") or 0.0) for item in candidates.values()]
    min_raw = min(raw_scores)
    max_raw = max(raw_scores)
    spread = max(max_raw - min_raw, 1.0)

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for path, item in candidates.items():
        snippet = str(item.get("snippet") or "")
        raw_score = float(item.get("score") or 0.0)
        keyword_score = max(0.0, min(1.0, (raw_score - min_raw) / spread))
        coverage, matched_terms, total_terms = _term_coverage_score(path, snippet, primary_terms)
        source_priority = _path_source_priority(path)
        intent_bonus = _query_intent_bonus(path, query)
        penalty = _noise_penalty(path, query)
        semantic_score = float(item.get("semantic_score") or 0.0)

        final_score = (
            0.35 * semantic_score
            + 0.30 * keyword_score
            + 0.20 * source_priority
            + 0.10 * coverage
            + 0.05 * intent_bonus
            - penalty
        )
        if coverage == 1.0 and total_terms:
            final_score += 0.12
        elif total_terms and coverage < 0.5:
            final_score -= 0.10

        enriched = dict(item)
        enriched["score"] = round(raw_score, 4)
        enriched["final_score"] = round(final_score, 4)
        enriched["score_breakdown"] = {
            "semantic": round(semantic_score, 4),
            "keyword": round(keyword_score, 4),
            "source_priority": round(source_priority, 4),
            "term_coverage": round(coverage, 4),
            "matched_terms": matched_terms,
            "total_terms": total_terms,
            "intent_bonus": round(intent_bonus, 4),
            "noise_penalty": round(penalty, 4),
            "raw_score": round(raw_score, 4),
        }
        ranked.append((final_score, path, enriched))

    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _score, _path, item in ranked[:limit]]


def _search_in_paths(
    paths: list[Path],
    vault: Path,
    terms: list[str],
    limit: int,
    max_chars: int,
    seen: set[str],
    *,
    query: str = "",
    primary_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    primary_terms = primary_terms or terms
    for path in paths:
        if not path.is_file():
            continue
        rel = str(path.relative_to(vault))
        if rel in seen:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        low = _normalize_search_text(text)
        name = _normalize_search_text(path.name)
        normalized_terms = [_normalize_search_text(t) for t in terms]
        if not any(t in low or t in name for t in normalized_terms):
            continue
        first = min((low.find(t) for t in normalized_terms if t in low), default=0)
        start = max(0, first - 160)
        snippet = text[start:start + max_chars].strip()
        wikilinks_seen: set[str] = set()
        wikilinks: list[str] = []
        for match in _WIKILINK_RE.finditer(text):
            link = match.group(1).strip()
            if link and link not in wikilinks_seen:
                wikilinks_seen.add(link)
                wikilinks.append(link)
        hit = {
            "path": rel,
            "snippet": snippet,
            "wikilinks": wikilinks,
            "score": _candidate_score(
                path=rel,
                text=text,
                primary_terms=primary_terms,
                expanded_terms=terms,
                query=query,
            ),
        }
        candidates.append(hit)
    candidates.sort(key=lambda item: (-float(item["score"]), item["path"]))
    selected = candidates[:limit]
    seen.update(str(item["path"]) for item in selected)
    return selected


_CHAT_LOG_DIRS = ("AI Chat", "Improvement Log", "Weekly Review", "Daily")
_PRIVATE_NOTE_DIRS = ("Private Reflection",)


def _is_chat_log(path: Path) -> bool:
    parts = path.parts
    return any(d in parts for d in _CHAT_LOG_DIRS)


def _is_private_note(path: Path) -> bool:
    """Notes readable by the user but excluded from every AI retrieval path."""
    return any(directory in path.parts for directory in _PRIVATE_NOTE_DIRS)


# モジュール起動時に1回だけ vault を走査し、Flask の request thread での rglob 不安定挙動を回避する。
_VAULT_INDEX: dict[str, Any] = {
    "vault": None,
    "knowledge_paths": [],  # チャットログ以外の .md
    "chat_log_paths": [],
    "built_at": 0.0,
}
_VAULT_INDEX_TTL_SEC = 300  # 5分


def _build_vault_index() -> None:
    """Vault配下の .md を走査し、知識ノートとチャットログに振り分けてキャッシュ。"""
    vault = find_vault()
    if not vault:
        _VAULT_INDEX.update(vault=None, knowledge_paths=[], chat_log_paths=[], built_at=time.time())
        return
    knowledge: list[Path] = []
    chat_logs: list[Path] = []
    try:
        for p in vault.rglob("*.md"):
            if _is_private_note(p):
                continue
            if _is_chat_log(p):
                chat_logs.append(p)
            else:
                knowledge.append(p)
    except OSError:
        pass
    _VAULT_INDEX.update(
        vault=vault,
        knowledge_paths=knowledge,
        chat_log_paths=chat_logs,
        built_at=time.time(),
    )


def _get_indexed_paths() -> tuple[Path | None, list[Path], list[Path]]:
    """TTL切れなら再構築してキャッシュを返す。"""
    if time.time() - _VAULT_INDEX["built_at"] > _VAULT_INDEX_TTL_SEC:
        _build_vault_index()
    return (
        _VAULT_INDEX["vault"],
        list(_VAULT_INDEX["knowledge_paths"]),
        list(_VAULT_INDEX["chat_log_paths"]),
    )


def iter_indexed_obsidian_documents(
    *,
    include_chat_logs: bool = False,
    max_chars: int = 1000,
) -> list[dict[str, str]]:
    """Return indexed Obsidian documents through the shared bridge path.

    AI chat and RAG entry points should use this helper instead of scanning the
    vault directly, so folder prioritization and exclusions stay consistent.
    """
    vault, knowledge, chat_logs = _get_indexed_paths()
    if not vault:
        return []
    paths = knowledge + (chat_logs if include_chat_logs else [])
    documents: list[dict[str, str]] = []
    for path in paths:
        if (
            not path.is_file()
            or _is_private_note(path)
            or any(skip in path.parts for skip in (".obsidian", ".claude", ".claudian"))
        ):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(path.relative_to(vault))
        except OSError:
            continue
        documents.append(
            {
                "title": path.stem,
                "path": rel_path,
                "content": content[:max_chars],
                "full_path": str(path),
                "source_type": "chat_log" if _is_chat_log(path) else "knowledge",
            }
        )
    return documents


# モジュール import 時にインデックスを初期化
_build_vault_index()


def search_notes(query: str, limit: int = 4, max_chars: int = 700) -> list[dict[str, str]]:
    """Search the shared Vault and rerank candidates by query relevance."""
    vault, knowledge, chat_logs = _get_indexed_paths()
    primary_terms = _split_query_terms(query)
    terms = _expand_query_terms(query)[:24]
    if not terms:
        return []

    candidates: dict[str, dict[str, Any]] = {}

    # Vector search supplies semantic candidates; lexical/path scoring decides final order.
    try:
        from obsidian_bridge_enhancements import get_vector_store_with_retry
        store = get_vector_store_with_retry()
        vector_limit = max(limit * 6, 20)
        for rank, item in enumerate(store.search(" ".join(terms), top_k=vector_limit)):
            raw_path = str(item.get("file_path") or "").strip()
            path = ""
            if raw_path:
                try:
                    raw = Path(raw_path)
                    if vault and raw.is_absolute():
                        try:
                            path = str(raw.relative_to(vault))
                        except ValueError:
                            continue
                    else:
                        path = raw_path
                except Exception:
                    path = raw_path
            if not path:
                path = str(item.get("file_name") or item.get("ref") or "").strip()
            if not path:
                continue
            if _is_private_note(Path(path)):
                continue
            text = str(item.get("text") or "").strip()
            semantic_score = _semantic_score_from_item(item, rank)
            candidates[path] = {
                "path": path,
                "snippet": text[:max_chars],
                "wikilinks": [
                    link.strip()
                    for link in str(item.get("wikilinks") or "").split(",")
                    if link.strip()
                ],
                "source": str(item.get("source") or "rag"),
                "semantic_score": round(semantic_score, 4),
                "vector_rank": rank + 1,
                "vector_distance": item.get("distance"),
                "vector_rank_score": item.get("rank_score"),
                "score": _candidate_score(
                    path=path,
                    text=text,
                    primary_terms=primary_terms,
                    expanded_terms=terms,
                    query=query,
                    source_bonus=max(2.0, 10.0 - rank * 0.5),
                ),
            }
    except Exception as e:
        import logging
        logging.debug(f"Vector store search failed: {e}, falling back to keyword search")

    if not vault:
        return _rerank_obsidian_candidates(
            candidates,
            query=query,
            primary_terms=primary_terms,
            limit=limit,
        )

    keyword_hits = _search_in_paths(
        knowledge + chat_logs,
        vault,
        terms,
        max(limit * 8, 30),
        max_chars,
        set(),
        query=query,
        primary_terms=primary_terms,
    )
    for hit in keyword_hits:
        path = str(hit["path"])
        existing = candidates.get(path)
        if existing is None or float(hit["score"]) > float(existing["score"]):
            merged = dict(hit)
            if existing:
                merged["semantic_score"] = existing.get("semantic_score", 0.0)
                merged["vector_rank"] = existing.get("vector_rank")
                merged["vector_distance"] = existing.get("vector_distance")
                merged["vector_rank_score"] = existing.get("vector_rank_score")
                merged["source"] = f"{existing.get('source', 'rag')}+keyword"
            else:
                merged["semantic_score"] = 0.0
                merged["source"] = "keyword"
            candidates[path] = merged
        elif existing is not None:
            existing["source"] = f"{existing.get('source', 'rag')}+keyword"

    return _rerank_obsidian_candidates(
        candidates,
        query=query,
        primary_terms=primary_terms,
        limit=limit,
    )


def _split_query_terms(query: str) -> list[str]:
    return split_query_terms(query)


def _expand_query_terms(query: str) -> list[str]:
    raw_terms = _split_query_terms(query)
    if not raw_terms:
        raw_terms = [query.lower().strip()]
    expanded = list(raw_terms)
    joined = " ".join(raw_terms)
    if any(k in joined for k in ("条件", "承認", "承認条件", "条件付", "条件付き")):
        expanded.extend([
            "承認条件",
            "条件付き承認",
            "条件付承認",
            "再提出",
            "保証",
            "担保",
            "期間短縮",
            "前受",
            "稟議",
            "q_risk",
            "信用リスク",
        ])
    if any(k in joined for k in ("obsidian", "保存", "メモ", "案件")):
        expanded.extend(["Projects/tune_lease_55/AI Chat", "Daily"])
    if any(k in joined for k in ("補助金", "助成金", "ものづくり", "省力化")):
        expanded.extend([
            "補助金",
            "助成金",
            "ものづくり補助金",
            "省力化投資補助金",
            "中小企業省力化投資補助金",
        ])
    domain_expansions = {
        "資金繰り": ["キャッシュフロー", "手元資金", "支払能力", "返済能力", "格付8-2", "格付8−2"],
        "銀行借入": ["借入", "銀行", "融資", "リースvs銀行借入"],
        "再リース": ["満了", "満了後", "返却", "買取", "再リース"],
        "残価": ["残存価値", "換金性", "中古売却", "処分"],
        "中古売却": ["中古相場", "換金性", "残存価値", "処分"],
        "動産保険": ["保険", "免責", "保険金", "事故"],
        "期待使用期間": ["経済耐用年数", "リース期間", "使用期間"],
        "建設業": ["業種別リースリスク", "業種別審査", "工事", "受注", "工期"],
    }
    scoring_expansions = {
        "scoring_core": ["審査ロジック", "スコアリング", "最終スコア", "判定基準"],
        "asset_score": ["物件スコア", "物件評価", "残価", "換金性", "担保価値"],
        "score_borrower": ["借手スコア", "借手評価", "信用力", "返済能力"],
        "統合": ["総合判断", "最終スコア", "合算", "補正"],
        "重み付け": ["寄与", "配点", "係数", "加点", "減点"],
        "物件スコア": ["asset_score", "物件評価", "残価", "換金性", "担保価値"],
        "借手スコア": ["score_borrower", "借手評価", "信用力", "返済能力"],
    }
    for trigger, aliases in domain_expansions.items():
        if trigger in joined:
            expanded.extend(aliases)
    for trigger, aliases in scoring_expansions.items():
        if trigger in joined:
            expanded.extend(aliases)

    # Cases/ 専用: 業種・スコア・判定キーワードで過去案件ログを引く
    _INDUSTRY_MAP = {
        "製造": "c 製造業", "建設": "d 建設業", "卸売": "i 卸売業",
        "小売": "j 小売業", "運輸": "h 運輸業", "情報": "g 情報通信業",
        "医療": "p 医療", "福祉": "p 福祉", "飲食": "m 宿泊業",
        "不動産": "l 不動産業", "サービス": "r サービス業",
    }
    for jp, cat in _INDUSTRY_MAP.items():
        if jp in joined:
            expanded.extend([cat, "cases/", f"cases/{dt.date.today().strftime('%Y-%m')}"])
            break
    if any(k in joined for k in ("スコア", "score", "高い", "低い", "承認", "否認", "否決", "要注意", "過去", "前回", "類似")):
        expanded.extend([
            "cases/",
            f"cases/{dt.date.today().strftime('%Y-%m')}",
            "スコア", "判定", "推奨金利", "q-risk",
        ])
    if any(k in joined for k in ("金利", "推奨", "レート", "spread", "rate")):
        expanded.extend(["推奨金利", "cases/", "recommended_rate"])
    if any(k in joined for k in ("q_risk", "q-risk", "qリスク", "財務矛盾", "量子")):
        expanded.extend(["q-risk", "cases/", "quantum_risk"])
    seen: set[str] = set()
    result: list[str] = []
    for term in expanded:
        term = term.strip().lower()
        if len(term) < 2 or term in seen:
            continue
        result.append(term)
        seen.add(term)
    return result


def recent_notes(limit: int = 3, folders: Iterable[str] | None = None, max_chars: int = 700) -> list[dict[str, str]]:
    vault, knowledge, chat_logs = _get_indexed_paths()
    if not vault:
        return []
    ym = dt.date.today().strftime("%Y-%m")
    folders = list(folders or (
        f"Projects/tune_lease_55/Cases/{ym}",
        "Projects/tune_lease_55/AI Chat",
        "Daily",
    ))
    indexed_paths = knowledge + chat_logs
    candidates: list[Path] = []
    for folder in folders:
        base = vault / folder
        candidates.extend(p for p in indexed_paths if base == p or base in p.parents)
    if not candidates:
        candidates = sorted(indexed_paths, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    seen: set[str] = set()
    hits: list[dict[str, str]] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(vault))
        if rel in seen:
            continue
        seen.add(rel)
        snippet = text.strip()[:max_chars]
        hits.append({"path": rel, "snippet": snippet})
        if len(hits) >= limit:
            break
    return hits


def collect_obsidian_context(query: str, limit: int = 4) -> list[dict[str, str]]:
    return search_notes(query, limit=limit)


def search_notes_with_industry_filter(
    query: str, industry_code: str | None = None, limit: int = 4
) -> list[dict[str, str]]:
    """【改善1】業種フィルタ付き検索。

    Args:
        query: 検索クエリ
        industry_code: 業種コード（例：'c' = 製造業）
        limit: 結果数上限

    Returns:
        メタデータ付きヒット
    """
    hits = search_notes(query, limit=limit * 2)  # 多めに取得してフィルタ
    if not industry_code:
        return hits[:limit]
    
    from obsidian_bridge_enhancements import filter_by_industry
    filtered = filter_by_industry(
        [{"path": h["path"], "metadata": {"industry": industry_code}} for h in hits],
        industry_code,
    )
    return hits[:limit] if not filtered else [h for h in hits if h["path"] in [f["path"] for f in filtered]]


def search_cases_by_score_range(
    query: str, min_score: float, max_score: float, limit: int = 4
) -> list[dict[str, str]]:
    """【改善2】スコア範囲で過去案件を検索。

    Cases/ フォルダ内で、frontmatter のスコア範囲が現在の案件と
    重なるノートを優先的に返す。

    Args:
        query: 検索クエリ（例：製造、建設）
        min_score: 現在の案件スコア下限
        max_score: 現在の案件スコア上限
        limit: 結果数上限

    Returns:
        関連度の高い過去案件
    """
    vault, _, _ = _get_indexed_paths()
    if not vault:
        return []
    
    cases_dir = vault / "Projects" / "tune_lease_55" / "Cases"
    case_notes = []
    try:
        for p in cases_dir.rglob("*.md"):
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if query.lower() in text.lower():
                        case_notes.append({
                            "path": str(p.relative_to(vault)),
                            "snippet": text[:700],
                            "metadata": {"score_range": (min_score, max_score)},
                        })
                except OSError:
                    pass
    except OSError:
        pass

    from obsidian_bridge_enhancements import filter_by_score_range
    filtered = filter_by_score_range(case_notes, min_score, max_score)
    return filtered[:limit]


def search_with_wikilink_context(query: str, limit: int = 4) -> list[dict[str, Any]]:
    """【改善3】Wikilink トラバーサル付き検索。

    検索結果に加えて、リンク先ノートの内容もコンテキストに含める。

    Args:
        query: 検索クエリ
        limit: 結果数上限

    Returns:
        {
            "path": "...",
            "snippet": "...",
            "wikilinks": ["[[link1]]", "..."],
            "linked_context": {"[[link1]]": "コンテンツ（500文字）"},
        }
    """
    hits = search_notes(query, limit=limit)
    vault, _, _ = _get_indexed_paths()
    if not vault:
        return hits

    from obsidian_bridge_enhancements import prefetch_wikilinks

    result = []
    for hit in hits:
        path = vault / hit["path"]
        linked_context = {}
        if path.exists():
            linked_content = prefetch_wikilinks(path, vault, max_depth=1)
            linked_context = linked_content

        result.append({
            **hit,
            "linked_context": linked_context,
        })

    return result


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_relevant_excerpt(snippet: str, terms: list[str], max_len: int = 240) -> str:
    text = (snippet or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:max_len]
    for line in lines:
        low = line.lower()
        if any(term in low for term in terms):
            return line[:max_len]
    bullet_lines = [line for line in lines if line.startswith(("-", "・", "*", "1.", "2.", "3.", "4.", "5."))]
    if bullet_lines:
        return bullet_lines[0][:max_len]
    return lines[0][:max_len]


def build_obsidian_digest(query: str, hits: list[dict[str, str]], max_items: int = 4) -> dict[str, str]:
    terms = _expand_query_terms(query)[:8]
    if not terms:
        terms = [_normalize_text(query)]
    dedup_paths: list[str] = []
    dedup_excerpts: list[str] = []
    wikilinks: list[str] = []
    seen_paths: set[str] = set()
    seen_excerpts: set[str] = set()
    for hit in hits[:max_items]:
        path = str(hit.get("path", "")).strip()
        if not path or path in seen_paths:
            continue
        excerpt = _extract_relevant_excerpt(str(hit.get("snippet", "")), terms)
        if not excerpt:
            continue
        seen_paths.add(path)
        dedup_paths.append(path)
        wikilinks.append(_to_wikilink(path, Path(path).stem))
        if excerpt not in seen_excerpts:
            seen_excerpts.add(excerpt)
            dedup_excerpts.append(excerpt)

    if not dedup_excerpts:
        return {"digest": "", "title": "Obsidian統合要約", "source_count": "0"}

    lines = ["## Obsidian統合要約", ""]
    lines.append(f"- 関連ノート数: {len(dedup_paths)}")
    if dedup_paths:
        lines.append(f"- 対象: {', '.join(dedup_paths[:max_items])}")
    lines.append("")
    lines.append("### 共通して見える要点")
    for excerpt in dedup_excerpts[:max_items]:
        lines.append(f"- {excerpt}")
    if wikilinks:
        lines.append("")
        lines.append("### 関連ノート")
        for link in wikilinks[:max_items]:
            lines.append(f"- {link}")
    return {
        "digest": "\n".join(lines).strip(),
        "title": "Obsidian統合要約",
        "source_count": str(len(dedup_paths)),
        "links": "\n".join(wikilinks[:max_items]),
    }


def _source_metadata_lines(
    *,
    source_query: str | None = None,
    related_paths: Iterable[str] | None = None,
) -> list[str]:
    source_notes: list[str] = []
    used_wiki_pages: list[str] = []
    for item in related_paths or []:
        rel = str(item or "").strip().replace("\\", "/")
        if not rel:
            continue
        source_notes.append(rel)
        if not _is_chat_log(Path(rel)):
            used_wiki_pages.append(rel)

    if not source_query and not source_notes:
        return []

    lines = ["", "### Source Metadata"]
    if source_query:
        safe_query = str(source_query).strip().replace("\n", " ")[:240]
        lines.append(f"- generated_from_query: {safe_query}")
    if used_wiki_pages:
        lines.append("- used_wiki_pages:")
        for rel in used_wiki_pages[:12]:
            lines.append(f"  - {rel}")
    if source_notes:
        lines.append("- source_notes:")
        for rel in source_notes[:12]:
            lines.append(f"  - {rel}")
    return lines


def append_chat_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AIチャットメモ").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntags: [チャット, AIチャット]\n---\n\n"
    else:
        header = "\n"
    section = f"## {now} {clean_title}\n\n### 要点\n{clean_body}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_improvement_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/Improvement Log/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AI改善候補").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntags: [チャット, 改善メモ]\n---\n\n"
    else:
        header = "\n"
    section = f"## {now} {clean_title}\n\n### 要点\n{clean_body}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_web_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/AI Chat/Web Research/{day}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "Web参照メモ").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntags: [チャット, Web参照]\n---\n\n"
    else:
        header = "\n"
    section = f"## {now} {clean_title}\n\n### 要点\n{clean_body}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_wiki_note(
    title: str,
    body: str,
    related_paths: Iterable[str] | None = None,
    source_query: str | None = None,
) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    rel = "Projects/tune_lease_55/tune_lease_55 Wiki.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "AI Wiki連携").strip()[:80]
    clean_body = (body or "").strip()
    related = []
    for item in related_paths or []:
        item = str(item or "").strip()
        if item:
            related.append(_to_wikilink(item, Path(item).stem))
    if not clean_body and not related:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        day = dt.date.today().isoformat()
        header = f"---\ndate: {day}\ntags: [知識, Wiki]\nsource: AI生成\n---\n\n"
    else:
        header = "\n"
    section_lines = [f"## {now} {clean_title}", ""]
    if related:
        section_lines.append("### 関連ノート")
        for link in related[:8]:
            section_lines.append(f"- {link}")
        section_lines.append("")
    section_lines.extend(_source_metadata_lines(source_query=source_query, related_paths=related_paths))
    if len(section_lines) > 2 and section_lines[-1] != "":
        section_lines.append("")
    if clean_body:
        section_lines.append(clean_body)
    section = "\n".join(section_lines).rstrip() + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_case_log(score_result: dict, case: dict) -> dict[str, str]:
    """スコアリング結果を Cases/ 以下の日次ログに追記する。"""
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}

    today = dt.date.today()
    ym = today.strftime("%Y-%m")
    ymd = today.isoformat()
    rel = f"Projects/tune_lease_55/Cases/{ym}/{ymd}.md"
    path = _safe_note_path(vault, rel)

    now = dt.datetime.now().strftime("%H:%M")

    industry = str(case.get("industry") or case.get("industry_major") or "不明")[:30]
    company = str(case.get("company_name") or "").strip()[:30]
    asset = str(case.get("asset_name") or "").strip()[:30]

    score = score_result.get("score") or (score_result.get("streamlit") or {}).get("score") or 0
    judgment = (
        score_result.get("judgment")
        or (score_result.get("streamlit") or {}).get("hantei")
        or "—"
    )
    recommended_rate = score_result.get("recommended_rate") or 0
    quantum_risk = (
        score_result.get("quantum_risk")
        or (score_result.get("aurion") or {}).get("quantum_risk")
        or 0
    )

    advisor = score_result.get("advisor") or {}
    advisor_summary = str(advisor.get("summary") or "").strip()[:120]
    risk_points = advisor.get("risk_points") or []
    main_risk = str(risk_points[0]).strip()[:80] if risk_points else "—"

    heading = f"## {now} | {industry} | スコア{score} | {judgment}"

    lines = [heading, ""]
    if company:
        lines.append(f"- **会社名**: {company}")
    if asset:
        lines.append(f"- **物件**: {asset}")
    lines.append(f"- **推奨金利**: {recommended_rate:.2f}%")
    lines.append(f"- **Q-Risk**: {quantum_risk}")
    lines.append(f"- **主リスク**: {main_risk}")
    if advisor_summary:
        lines.append(f"- **軍師サマリー**: {advisor_summary}")

    section = "\n".join(lines).rstrip() + "\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path)}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def append_asset_finance_note(
    asset_case: dict,
    result: dict,
    related_paths: Iterable[str] | None = None,
) -> dict[str, str]:
    """物件ファイナンス審査結果を Asset Finance/ 以下の日次ログに追記する。"""
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}

    today = dt.date.today()
    ym = today.strftime("%Y-%m")
    ymd = today.isoformat()
    rel = f"Projects/tune_lease_55/Asset Finance/{ym}/{ymd}.md"
    path = _safe_note_path(vault, rel)

    now = dt.datetime.now().strftime("%H:%M")
    asset_type = str(asset_case.get("asset_type") or "不明")[:30]
    asset_name = str(asset_case.get("asset_name") or "").strip()[:60]
    term = asset_case.get("term") or 0
    down_payment = asset_case.get("down_payment") or 0
    financial_score = str(asset_case.get("financial_score") or "不明")

    score = result.get("score", 0)
    decision = str(result.get("decision") or "—")
    bep_month = result.get("bep_month", "—")
    bep_ratio = result.get("bep_ratio", 0)
    reasons = [str(x).strip() for x in result.get("reasons") or [] if str(x).strip()]
    deductions = [str(x).strip() for x in result.get("deductions") or [] if str(x).strip()]
    action_plan = [str(x).strip() for x in result.get("action_plan") or [] if str(x).strip()]

    title_asset = asset_name or asset_type
    heading = f"## {now} | {title_asset} | {decision} | スコア{score}"
    lines = [heading, ""]
    lines.append(f"- **物件種別**: {asset_type}")
    if asset_name:
        lines.append(f"- **物件名**: {asset_name}")
    lines.append(f"- **期間**: {term}ヶ月")
    lines.append(f"- **自己資金率**: {_safe_float(down_payment) * 100:.0f}%")
    lines.append(f"- **財務評価**: {financial_score}")
    lines.append(f"- **BEP**: {bep_month}ヶ月目（期間比 {_safe_float(bep_ratio) * 100:.0f}%）")

    if related_paths:
        related_links = []
        for item in related_paths:
            item = str(item or "").strip()
            if item:
                related_links.append(_to_wikilink(item, Path(item).stem))
        if related_links:
            lines.append("")
            lines.append("### 関連メモ")
            lines.extend(f"- {link}" for link in related_links[:8])
            lines.extend(_source_metadata_lines(related_paths=related_paths))

    if reasons:
        lines.append("")
        lines.append("### 承認根拠")
        lines.extend(f"- {item}" for item in reasons[:8])
    if deductions:
        lines.append("")
        lines.append("### 減点・リスク")
        lines.extend(f"- {item}" for item in deductions[:8])
    if action_plan:
        lines.append("")
        lines.append("### 営業アクション")
        lines.extend(f"- {item}" for item in action_plan[:5])

    section = "\n".join(lines).rstrip() + "\n"
    prefix = "\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + section)
    return {"status": "saved", "path": str(path), "rel_path": rel}


def append_asset_knowledge_backlinks(
    asset_case: dict,
    result: dict,
    related_paths: Iterable[str] | None = None,
    finance_note_rel: str | None = None,
) -> dict[str, Any]:
    """Asset Knowledgeノート側へ、この知識を使った審査結果リンクを追記する。"""
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません", "updated": []}

    updated: list[str] = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    asset_type = str(asset_case.get("asset_type") or "不明")[:30]
    asset_name = str(asset_case.get("asset_name") or asset_type).strip()[:60]
    score = result.get("score", 0)
    decision = str(result.get("decision") or "—")
    finance_link = _to_wikilink(finance_note_rel or "", "審査結果") if finance_note_rel else ""

    for item in related_paths or []:
        rel = str(item or "").strip().replace("\\", "/")
        if not rel.startswith("Projects/tune_lease_55/Asset Knowledge/"):
            continue
        path = _safe_note_path(vault, rel)
        if not path.exists():
            continue
        lines = [
            f"## {now} 使用案件",
            "",
            f"- **物件**: {asset_name}",
            f"- **物件種別**: {asset_type}",
            f"- **判定**: {decision}",
            f"- **スコア**: {score}",
        ]
        if finance_link:
            lines.append(f"- **審査ログ**: {finance_link}")
        section = "\n".join(lines).rstrip() + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + section)
        updated.append(rel)

    return {"status": "saved", "updated": updated}


def append_work_log(
    title: str,
    what: str,
    why_hard: str = "",
    next_time: str = "",
    lesson: str = "",
    pr: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, str]:
    """Codexスタイルの作業ログをObsidianに追記する。"""
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    day = dt.date.today().isoformat()
    rel = f"Projects/tune_lease_55/Work Logs/{day}.md"
    path = _safe_note_path(vault, rel)
    _tags = tags or ["作業ログ"]
    tag_str = ", ".join(_tags)
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntype: work_log\ntags: [{tag_str}]\n---\n\n"
    else:
        header = "\n"
    pr_suffix = f"（PR #{pr}）" if pr else ""
    section_lines = [
        f"## 作業: {title}{pr_suffix}",
        "",
        "### 何をしたか",
        (what or "").strip(),
    ]
    if why_hard:
        section_lines += ["", "### なぜ大変だったか", why_hard.strip()]
    if next_time:
        section_lines += ["", "### 次回どう切り分けるか", next_time.strip()]
    if lesson:
        section_lines += ["", "### 教訓", lesson.strip()]
    section = "\n".join(section_lines) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_weekly_review_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    iso_year, iso_week, _ = dt.date.today().isocalendar()
    rel = f"Projects/tune_lease_55/AI Chat/Weekly Review/{iso_year}-W{iso_week:02d}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "週次改善レビュー").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntags: [チャット, 週次レビュー]\n---\n\n"
    else:
        header = "\n"
    section = f"## {now} {clean_title}\n\n### 要点\n{clean_body}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}


def append_monthly_review_note(title: str, body: str) -> dict[str, str]:
    vault = find_vault()
    if not vault:
        return {"status": "skipped", "reason": "iCloud 上の Obsidian Vault が見つかりません"}
    year_month = dt.date.today().strftime("%Y-%m")
    rel = f"Projects/tune_lease_55/AI Chat/Monthly Review/{year_month}.md"
    path = _safe_note_path(vault, rel)
    now = dt.datetime.now().strftime("%H:%M")
    clean_title = (title or "月次改善レビュー").strip()[:80]
    clean_body = (body or "").strip()
    if not clean_body:
        return {"status": "skipped", "reason": "empty note body"}
    is_new = not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip()
    if is_new:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"---\ndate: {timestamp}\ntags: [チャット, 月次レビュー]\n---\n\n"
    else:
        header = "\n"
    section = f"## {now} {clean_title}\n\n### 要点\n{clean_body}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + section)
    return {"status": "saved", "path": str(path)}
