"""
ChromaDB 永続化ベクトルストア。
paraphrase-multilingual-MiniLM-L12-v2 モデルで日本語テキストをベクトル化する。
"""
from __future__ import annotations

import datetime
import os
import hashlib
import json
import logging
import math
import re
import threading
from typing import Literal

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
_COLLECTION_NAME = "obsidian_knowledge"
_RANKING_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "rag_ranking.json")
_REMOTE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_LOCAL_MODEL_DIR = os.path.join(
    _REPO_ROOT,
    "models",
    "sentence-transformers",
    _REMOTE_MODEL_NAME,
)
_MODEL_NAME = os.environ.get("OBSIDIAN_RAG_MODEL") or (
    _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else _REMOTE_MODEL_NAME
)

_PREFERRED_PATH_BOOSTS = (
    ("knowledge_base/okf_lease_concepts/", 0.10),
    ("リース知識/", 0.10),
    ("03-知識_業界/", 0.09),
    ("Projects/tune_lease_55/Asset Knowledge/", 0.09),
    ("Projects/tune_lease_55/Asset Finance/", 0.09),
    ("Projects/tune_lease_55/Cases/", 0.08),
    ("Projects/tune_lease_55/Feedback/", 0.07),
    ("Projects/tune_lease_55/Research/", 0.07),
    ("Lease Intelligence/Memory/", 0.06),
    ("Projects/tune_lease_55/", 0.05),
    ("tuneLease55/知見・分析/", 0.05),
)
_LOW_PRIORITY_PATH_PENALTIES = (
    ("05-クリップ_記事/リースニュース/", 0.25),
    ("リースニュース/", 0.25),
    ("07-アーカイブ/", 0.16),
    ("Projects/tune_lease_55/AI Chat/", 0.15),
    ("Projects/tune_lease_55/Improvement Log/", 0.15),
    ("Projects/tune_lease_55/Weekly Review/", 0.15),
    ("Daily/", 0.12),
    ("Clippings/", 0.08),
    ("Humor/", 0.05),
)
_CONTEXTUAL_NOISE_TERMS = ("八奈見", "キャラクター", "ユーモア", "口調", "Humor")
_NOISE_ALLOWED_TERMS = ("八奈見", "キャラ", "ユーモア", "口調", "冗談", "笑", "yanami", "humor")

_DEFAULT_RANKING_CONFIG = {
    "preferred_path_boosts": dict(_PREFERRED_PATH_BOOSTS),
    "low_priority_path_penalties": dict(_LOW_PRIORITY_PATH_PENALTIES),
    "sync_copy_penalty": 0.35,
    "keyword_pool_multiplier": 4,
    "keyword_pool_min": 12,
}

_SEARCH_LOG_PATH = os.path.join(_REPO_ROOT, "data", "rag_search_log.jsonl")
_search_log_lock = threading.Lock()


def _write_search_log(query: str, surface: str, results: list[dict]) -> None:
    if not results:
        return
    try:
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "surface": surface,
            "results": [
                {
                    "rank": i + 1,
                    "doc_id": r.get("doc_id", ""),
                    "file_name": r.get("file_name", ""),
                    "obsidian_ref": r.get("ref", ""),
                    "final_score": r.get("rank_score"),
                    "score_breakdown": r.get("score_breakdown"),
                }
                for i, r in enumerate(results[:5])
            ],
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _search_log_lock:
            with open(_SEARCH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        logger.warning("[KnowledgeVectorStore] search log write failed: %s", exc)


def load_ranking_config(path: str = _RANKING_CONFIG_PATH) -> dict:
    config = {
        "preferred_path_boosts": dict(_DEFAULT_RANKING_CONFIG["preferred_path_boosts"]),
        "low_priority_path_penalties": dict(_DEFAULT_RANKING_CONFIG["low_priority_path_penalties"]),
        "sync_copy_penalty": _DEFAULT_RANKING_CONFIG["sync_copy_penalty"],
        "keyword_pool_multiplier": _DEFAULT_RANKING_CONFIG["keyword_pool_multiplier"],
        "keyword_pool_min": _DEFAULT_RANKING_CONFIG["keyword_pool_min"],
    }
    try:
        with open(path, encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except (OSError, ValueError, TypeError):
        return config
    if not isinstance(raw, dict):
        return config
    for key in ("preferred_path_boosts", "low_priority_path_penalties"):
        values = raw.get(key)
        if isinstance(values, dict):
            for prefix, value in values.items():
                try:
                    config[key][str(prefix)] = float(value)
                except (TypeError, ValueError):
                    continue
    for key in ("sync_copy_penalty",):
        try:
            config[key] = float(raw.get(key, config[key]))
        except (TypeError, ValueError):
            pass
    for key in ("keyword_pool_multiplier", "keyword_pool_min"):
        try:
            config[key] = int(raw.get(key, config[key]))
        except (TypeError, ValueError):
            pass
    return config


class KnowledgeVectorStore:
    """ChromaDB ラッパー。遅延初期化でスタートアップをブロックしない。"""

    def __init__(
        self,
        chroma_dir: str = _CHROMA_DIR,
        model_name: str = _MODEL_NAME,
        ranking_config: dict | None = None,
    ):
        self._chroma_dir = chroma_dir
        self._model_name = model_name
        self._client = None
        self._collection = None
        self._encoder = None
        self._encoder_failed = False
        self._ranking_config_path = _RANKING_CONFIG_PATH if ranking_config is None else None
        self._ranking_config_mtime = 0.0
        self._ranking_config = ranking_config or load_ranking_config()
        if self._ranking_config_path:
            try:
                self._ranking_config_mtime = os.path.getmtime(self._ranking_config_path)
            except OSError:
                pass
        self._init_lock = threading.Lock()
        self._encoder_lock = threading.Lock()

    def set_ranking_config(self, config: dict) -> None:
        """Replace only the bounded retrieval-ranking configuration."""
        self._ranking_config = config

    def _maybe_reload_ranking_config(self) -> None:
        if not self._ranking_config_path:
            return
        try:
            mtime = os.path.getmtime(self._ranking_config_path)
        except OSError:
            return
        if mtime <= self._ranking_config_mtime:
            return
        self._ranking_config = load_ranking_config(self._ranking_config_path)
        self._ranking_config_mtime = mtime
        logger.info("[KnowledgeVectorStore] ranking config reloaded: %s", self._ranking_config_path)

    def _preferred_path_boosts(self) -> tuple[tuple[str, float], ...]:
        values = self._ranking_config.get("preferred_path_boosts") or {}
        return tuple((str(prefix), float(value)) for prefix, value in values.items())

    def _low_priority_path_penalties(self) -> tuple[tuple[str, float], ...]:
        values = self._ranking_config.get("low_priority_path_penalties") or {}
        return tuple((str(prefix), float(value)) for prefix, value in values.items())

    def _ensure_collection(self) -> None:
        """初回アクセス時に ChromaDB collection だけを初期化する。"""
        if self._collection is not None:
            return

        with self._init_lock:
            if self._collection is not None:
                return

            import chromadb

            os.makedirs(self._chroma_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[KnowledgeVectorStore] collection initialized: %s", self._chroma_dir)

    def _ensure_encoder(self) -> bool:
        """ローカルキャッシュ済み encoder だけを読む。未キャッシュならネットへ出ず false。"""
        if self._encoder is not None:
            return True
        if self._encoder_failed:
            return False
        allow_remote_model_name = os.environ.get("OBSIDIAN_RAG_USE_ENCODER", "").strip() == "1"
        if not allow_remote_model_name and not os.path.isdir(self._model_name):
            # sentence-transformers/huggingface_hub can still issue network HEAD
            # requests for named models even with local_files_only. Default to
            # deterministic offline search unless the operator explicitly opts in.
            self._encoder_failed = True
            logger.info(
                "[KnowledgeVectorStore] encoder skipped; download the model to %s, "
                "set OBSIDIAN_RAG_MODEL to a local directory, or set OBSIDIAN_RAG_USE_ENCODER=1 "
                "to allow loading by remote model name",
                _LOCAL_MODEL_DIR,
            )
            return False

        with self._encoder_lock:
            if self._encoder is not None:
                return True
            if self._encoder_failed:
                return False
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    self._model_name,
                    device="cpu",
                    local_files_only=True,
                )
                logger.info("[KnowledgeVectorStore] encoder initialized locally: %s", self._model_name)
                return True
            except TypeError:
                # 古い sentence-transformers でも外部アクセスを避ける。
                old_hf = os.environ.get("HF_HUB_OFFLINE")
                old_tf = os.environ.get("TRANSFORMERS_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = SentenceTransformer(self._model_name, device="cpu")
                    logger.info("[KnowledgeVectorStore] encoder initialized in offline mode: %s", self._model_name)
                    return True
                except Exception as exc:
                    logger.warning("[KnowledgeVectorStore] local encoder unavailable: %s", exc)
                    self._encoder_failed = True
                    return False
                finally:
                    if old_hf is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_hf
                    if old_tf is None:
                        os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    else:
                        os.environ["TRANSFORMERS_OFFLINE"] = old_tf
            except Exception as exc:
                logger.warning("[KnowledgeVectorStore] local encoder unavailable: %s", exc)
                self._encoder_failed = True
                return False

    def _ensure_initialized(self) -> None:
        """互換用: collection を初期化し、encoder は使えれば読む。"""
        self._ensure_collection()
        self._ensure_encoder()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_collection()
        if not self._ensure_encoder():
            return [self._hash_embed(text) for text in texts]
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def _hash_embed(text: str, dimension: int = 384) -> list[float]:
        """Offline deterministic embedding used only when the ML encoder is unavailable."""
        vec = [0.0] * dimension
        tokens = re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]{2,}", (text or "").lower())
        if not tokens:
            tokens = [(text or "")[:64]]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def upsert_chunks(self, chunks: list) -> int:
        """Chunk リストを ChromaDB にアップサート。追加件数を返す。"""
        if not chunks:
            return 0
        self._ensure_collection()

        ids = [c.doc_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{
            "file_name": c.file_name,
            "file_path": c.file_path,
            "section": c.section,
            "obsidian_ref": c.obsidian_ref,
            "mtime": c.mtime,
            **{k: str(v) for k, v in c.metadata.items() if k not in ("section",)},
        } for c in chunks]
        embeddings = self._embed(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        try:
            from obsidian_query import split_query_terms

            terms = split_query_terms(query)
        except Exception:
            terms = re.split(r"[\s,、。.!！?？:：;；/／\\|()\[\]{}「」『』【】<>＜＞]+", (query or "").lower())
        result: list[str] = []
        seen: set[str] = set()
        for term in terms:
            t = str(term or "").strip().lower()
            if len(t) >= 2 and t not in seen:
                result.append(t)
                seen.add(t)
        return result

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """encoder が使えない環境向けの Chroma document キーワード検索。"""
        self._ensure_collection()
        terms = self._query_terms(query)
        if not terms or self._collection.count() == 0:
            return []
        try:
            result = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("[KnowledgeVectorStore] keyword fallback failed: %s", exc)
            return []

        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ranked: list[tuple[int, int, str, dict, str]] = []
        strong_terms = [term for term in terms if len(term) >= 3]
        for idx, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            text = str(doc or "")
            metadata = meta or {}
            file_name = str(metadata.get("file_name") or "").lower()
            section = str(metadata.get("section") or "").lower()
            ref = str(metadata.get("obsidian_ref") or "").lower()
            wikilinks = str(metadata.get("wikilinks") or "").lower()
            metadata_text = self._metadata_text(metadata).lower()
            haystack = " ".join(
                [
                    text,
                    file_name,
                    section,
                    ref,
                    wikilinks,
                    metadata_text,
                ]
            ).lower()
            if strong_terms and not any(term in haystack for term in strong_terms):
                continue
            score = 0
            for term in terms:
                if term in file_name:
                    score += 6
                elif term in section or term in ref:
                    score += 4
                elif term in wikilinks:
                    score += 2
                elif term in haystack:
                    score += 1
            if score:
                ranked.append((score, -idx, text, metadata, doc_id))

        hits = []
        for score, _idx, doc, meta, did in ranked:
            hits.append({
                "doc_id": did,
                "text": doc,
                "ref": meta.get("obsidian_ref", ""),
                "file_name": meta.get("file_name", ""),
                "file_path": meta.get("file_path", ""),
                "section": meta.get("section", ""),
                "wikilinks": meta.get("wikilinks", ""),
                "metadata": meta,
                "distance": None,
                "score": score,
                "source": "keyword_fallback",
            })
        return self._rerank_hits(query, hits, top_k=top_k)

    @staticmethod
    def _metadata_text(metadata: dict) -> str:
        values: list[str] = []
        for key in ("type", "title", "domain", "tags", "source", "confidence", "status", "related"):
            value = metadata.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value is not None:
                values.append(str(value))
        return " ".join(values)

    @staticmethod
    def _display_path(meta_or_hit: dict) -> str:
        raw_path = str(meta_or_hit.get("file_path") or "").replace("\\", "/")
        if "/Obsidian Vault/" in raw_path:
            raw_path = raw_path.split("/Obsidian Vault/", 1)[1]
        return raw_path or str(meta_or_hit.get("file_name") or meta_or_hit.get("ref") or "")

    @staticmethod
    def _allows_contextual_noise(query: str) -> bool:
        low = (query or "").lower()
        return any(term.lower() in low for term in _NOISE_ALLOWED_TERMS)

    def _business_priority(self, query: str, hit: dict) -> float:
        path = self._display_path(hit)
        file_name = str(hit.get("file_name") or "")
        section = str(hit.get("section") or "")
        ref = str(hit.get("ref") or "")
        haystack = f"{path} {file_name} {section} {ref}"

        if not self._allows_contextual_noise(query) and any(term in haystack for term in _CONTEXTUAL_NOISE_TERMS):
            return -10.0

        low_query = (query or "").lower()
        score = 0.0
        for prefix, boost in self._preferred_path_boosts():
            if path.startswith(prefix):
                score += boost
                break
        for prefix, penalty in self._low_priority_path_penalties():
            if path.startswith(prefix):
                score -= penalty
                break
        if "/lease-wiki-vault/" in path:
            score -= float(self._ranking_config.get("sync_copy_penalty", 0.35))

        case_intent = any(term in low_query for term in ("過去", "類似", "案件", "事例", "前回", "cases"))
        asset_intent = any(term in low_query for term in ("物件", "残価", "再販", "中古", "換金", "処分", "売却"))
        if path.startswith("Projects/tune_lease_55/Cases/") and not case_intent:
            score -= 0.13
        if path.startswith("Projects/tune_lease_55/Asset Knowledge/") and not asset_intent:
            score -= 0.08

        terms = self._query_terms(query)
        low_haystack = haystack.lower()
        exact_matches = sum(1 for term in terms if term in low_haystack)
        score += min(0.08, exact_matches * 0.02)

        if "検索語インデックス" in file_name and not any(term in query for term in ("検索語", "インデックス", "wiki")):
            score -= 0.20
        if "Wiki" in file_name or "wiki" in file_name:
            score -= 0.02
        if any(term in low_query for term in ("銀行借入", "違い", "比較", "営業説明")) and (
            "銀行借入" in file_name or "リースvs銀行借入" in file_name
        ):
            score += 0.12
        if "資金繰り" in low_query and any(term in file_name for term in ("審査実務", "業種別リースリスク", "格付")):
            score += 0.08
        if "建設業" in low_query and "業種別リースリスク" in file_name:
            score += 0.08
        if "再リース" in low_query and any(term in file_name for term in ("リース満了後", "再リース")):
            score += 0.10
        return score

    def _term_coverage(self, query: str, hit: dict) -> tuple[float, int, int]:
        terms = [term for term in self._query_terms(query) if term not in {"確認", "注意", "リスク", "関係", "方法"}]
        if not terms:
            return 0.0, 0, 0
        haystack = " ".join(
            [
                str(hit.get("text") or ""),
                str(hit.get("file_name") or ""),
                str(hit.get("section") or ""),
                str(hit.get("ref") or ""),
                self._display_path(hit),
                str(hit.get("wikilinks") or ""),
                self._metadata_text(hit.get("metadata") or {}),
            ]
        ).lower()
        matched = sum(1 for term in terms if term in haystack)
        return matched / len(terms), matched, len(terms)

    def _source_priority(self, hit: dict) -> float:
        path = self._display_path(hit)
        if "/lease-wiki-vault/" in path:
            return 0.15
        for prefix, boost in self._preferred_path_boosts():
            if path.startswith(prefix):
                return min(1.0, 0.72 + boost * 2.2)
        for prefix, penalty in self._low_priority_path_penalties():
            if path.startswith(prefix):
                return max(0.1, 0.50 - penalty)
        return 0.55

    def _noise_penalty(self, query: str, hit: dict) -> float:
        priority = self._business_priority(query, hit)
        if priority <= -9.0:
            return 1.0
        path = self._display_path(hit)
        file_name = str(hit.get("file_name") or "")
        low_query = (query or "").lower()
        penalty = 0.0
        if not any(term in low_query for term in ("チャット", "会話", "日報", "weekly", "改善ログ")):
            if any(path.startswith(prefix) for prefix, _penalty in self._low_priority_path_penalties()):
                penalty += 0.18
        if any(
            path.startswith(prefix)
            for prefix in ("05-クリップ_記事/リースニュース/", "リースニュース/")
        ) and "ニュース" not in low_query:
            penalty += 0.45
        if path.startswith("07-アーカイブ/"):
            penalty += 0.18
        if "/lease-wiki-vault/" in path:
            penalty += float(self._ranking_config.get("sync_copy_penalty", 0.35))
        if "検索語インデックス" in file_name and not any(term in query for term in ("検索語", "インデックス", "wiki")):
            penalty += 0.18
        if ("Wiki" in file_name or "wiki" in file_name) and "wiki" not in low_query:
            penalty += 0.04
        return min(penalty, 1.0)

    def _rerank_hits(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        ranked: list[tuple[float, int, dict]] = []
        for idx, hit in enumerate(hits):
            priority = self._business_priority(query, hit)
            if priority <= -9.0:
                continue
            if hit.get("distance") is not None:
                distance = float(hit.get("distance") or 0.0)
                semantic_score = max(0.0, min(1.0, 1.0 - distance))
                base = semantic_score
            else:
                semantic_score = 0.0
                base = min(1.0, float(hit.get("score") or 0.0) / 20.0)
            coverage, matched_terms, total_terms = self._term_coverage(query, hit)
            source_priority = self._source_priority(hit)
            noise_penalty = self._noise_penalty(query, hit)
            rank_score = (
                0.40 * base
                + 0.25 * coverage
                + 0.20 * source_priority
                + 0.15 * max(-0.3, min(0.3, priority))
                - noise_penalty
            )
            if coverage == 1.0 and total_terms:
                rank_score += 0.08
            elif total_terms and coverage < 0.5:
                rank_score -= 0.08
            item = dict(hit)
            item["rank_score"] = round(rank_score, 4)
            item["priority_score"] = round(priority, 4)
            item["score_breakdown"] = {
                "semantic": round(semantic_score, 4),
                "base": round(base, 4),
                "term_coverage": round(coverage, 4),
                "matched_terms": matched_terms,
                "total_terms": total_terms,
                "source_priority": round(source_priority, 4),
                "business_priority": round(priority, 4),
                "noise_penalty": round(noise_penalty, 4),
            }
            ranked.append((rank_score, -idx, item))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[dict] = []
        seen_paths: set[str] = set()
        for _score, _idx, item in ranked:
            path = self._display_path(item)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    def search(
        self,
        query: str,
        mode: Literal["support", "refute", "both"] = "both",
        top_k: int = 3,
        surface: str = "",
    ) -> list[dict]:
        """
        クエリに近いチャンクを検索する。

        Args:
            query:  検索クエリ（自然言語）
            mode:   "support" → 肯定的エビデンス、"refute" → 否定的エビデンス、
                    "both" → 両方（プレフィックスを付けて検索）
            top_k:  返す最大件数

        Returns:
            [{"text": str, "ref": str, "distance": float, ...}, ...]
        """
        self._maybe_reload_ranking_config()
        self._ensure_collection()

        if self._collection.count() == 0:
            return []

        if mode == "support":
            effective_query = f"成功事例 承認 リスク低い {query}"
        elif mode == "refute":
            effective_query = f"失敗事例 否決 リスク高い 問題 {query}"
        else:
            effective_query = query

        if not self._ensure_encoder():
            hits = self._keyword_search(effective_query, top_k)
            _write_search_log(query, surface, hits)
            return hits

        try:
            embedding = self._embed([effective_query])[0]
            candidate_count = min(max(top_k * 8, top_k + 8), max(1, self._collection.count()))
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("[KnowledgeVectorStore] vector search failed, falling back to keyword: %s", exc)
            hits = self._keyword_search(effective_query, top_k)
            _write_search_log(query, surface, hits)
            return hits

        hits = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "doc_id": doc_id,
                "text": doc,
                "ref": meta.get("obsidian_ref", ""),
                "file_name": meta.get("file_name", ""),
                "file_path": meta.get("file_path", ""),
                "section": meta.get("section", ""),
                "wikilinks": meta.get("wikilinks", ""),
                "metadata": meta,
                "distance": round(float(dist), 4),
                "source": "vector",
            })

        # Semantic search can miss exact domain terms in long notes. Merge a
        # lexical candidate pool before the final evaluator-owned rerank.
        keyword_multiplier = max(1, int(self._ranking_config.get("keyword_pool_multiplier", 4)))
        keyword_min = max(top_k, int(self._ranking_config.get("keyword_pool_min", 12)))
        keyword_hits = self._keyword_search(
            effective_query,
            top_k=max(top_k * keyword_multiplier, keyword_min),
        )
        merged: dict[tuple[str, str], dict] = {}
        for hit in [*hits, *keyword_hits]:
            key = (self._display_path(hit), str(hit.get("section") or ""))
            existing = merged.get(key)
            if existing is None:
                merged[key] = hit
                continue
            if existing.get("distance") is None and hit.get("distance") is not None:
                merged[key] = hit
            elif hit.get("score") is not None:
                existing["score"] = max(float(existing.get("score") or 0.0), float(hit.get("score") or 0.0))
                existing["source"] = "vector+keyword"
        final = self._rerank_hits(query, list(merged.values()), top_k=top_k)
        _write_search_log(query, surface, final)
        return final

    def count(self) -> int:
        """インデックス内のドキュメント数を返す。"""
        try:
            self._ensure_collection()
            return self._collection.count()
        except Exception:
            return 0


# モジュールレベルシングルトン（APIサーバーで共有）
_store: KnowledgeVectorStore | None = None


def get_store() -> KnowledgeVectorStore:
    global _store
    if _store is None:
        _store = KnowledgeVectorStore()
    return _store
