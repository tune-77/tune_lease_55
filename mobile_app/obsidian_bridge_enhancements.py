"""
Obsidian RAG 改善モジュール
- Frontmatter メタデータ抽出
- BM25 ランキング
- ファイルハッシュベース差分更新
- リトライロジック
- 業種・スコアフィルタ
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from functools import lru_cache

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args, **kwargs):
        return lambda f: f
    def stop_after_attempt(*args): pass
    def wait_exponential(*args): pass

logger = logging.getLogger(__name__)

# ============================================================================
# Frontmatter パーサ
# ============================================================================

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def extract_frontmatter(text: str) -> dict[str, Any]:
    """ノート冒頭の YAML frontmatter を抽出。

    Returns:
        メタデータ辞書。frontmatter がなければ {}
    """
    if not yaml:
        return {}
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}


def extract_metadata(path: Path, text: str) -> dict[str, Any]:
    """ファイルパス + Frontmatter からメタデータを抽出。

    Returns:
        {
            "path": "relative/path",
            "title": "...",
            "tags": ["tag1", "tag2"],
            "industry": "c 製造業",
            "score_range": (60, 80),
            "credit_rating": "4-6",
            "modified_at": 1234567890.0,
            "word_count": 150,
            "has_wikilinks": true,
        }
    """
    vault_root = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault"
    try:
        rel_path = path.relative_to(vault_root)
    except ValueError:
        rel_path = path

    fm = extract_frontmatter(text)

    # タグ抽出（frontmatter + body の #tag）
    tags = set(fm.get("tags") or [])
    body_tags = set(re.findall(r"#([a-zA-Z0-9_\-ぁ-ん]+)", text))
    tags.update(body_tags)

    # 業種マッピング
    industry = fm.get("industry") or fm.get("category") or None
    if isinstance(industry, str):
        if "製造" in industry or "c " in industry:
            industry = "c 製造業"
        elif "建設" in industry or "d " in industry:
            industry = "d 建設業"
        # ... その他の業種も同様

    # スコア範囲（frontmatter に min_score/max_score があれば）
    score_range = None
    if fm.get("min_score") is not None and fm.get("max_score") is not None:
        try:
            score_range = (int(fm["min_score"]), int(fm["max_score"]))
        except (ValueError, TypeError):
            score_range = None

    # 信用格付
    credit_rating = fm.get("credit_rating") or fm.get("credit_class") or None

    return {
        "path": str(rel_path),
        "title": fm.get("title") or path.stem,
        "tags": list(tags),
        "industry": industry,
        "score_range": score_range,
        "credit_rating": credit_rating,
        "modified_at": path.stat().st_mtime if path.exists() else 0.0,
        "word_count": len(text.split()),
        "has_wikilinks": bool(re.search(r"\[\[[^\]]+\]\]", text)),
    }


# ============================================================================
# ファイルハッシュベース差分更新
# ============================================================================

_FILE_HASH_CACHE: dict[str, tuple[str, float]] = {}  # path -> (hash, mtime)


def _file_hash(path: Path) -> str:
    """ファイルのSHA256ハッシュを計算。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _needs_update(path: Path) -> bool:
    """ファイルがキャッシュ以降に変更されたか判定。"""
    if path not in _FILE_HASH_CACHE:
        return True
    cached_hash, cached_mtime = _FILE_HASH_CACHE[path]
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        return True
    if current_mtime > cached_mtime:
        return True
    current_hash = _file_hash(path)
    return current_hash != cached_hash


def _update_hash_cache(path: Path) -> None:
    """ハッシュキャッシュを更新。"""
    try:
        mtime = path.stat().st_mtime
        fhash = _file_hash(path)
        _FILE_HASH_CACHE[path] = (fhash, mtime)
    except OSError:
        pass


def prune_stale_cache(vault: Path) -> None:
    """削除されたファイルをキャッシュから削除。"""
    stale = [p for p in _FILE_HASH_CACHE if not p.exists()]
    for p in stale:
        del _FILE_HASH_CACHE[p]


# ============================================================================
# BM25 ランキング
# ============================================================================

class BM25Scorer:
    """BM25 アルゴリズムでドキュメントをスコア付け。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.N = 0

    def fit(self, documents: list[str]) -> None:
        """ドキュメント群から IDF を学習。"""
        self.N = len(documents)
        doc_freq: dict[str, int] = {}
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.avgdl = total_len / max(1, self.N)

        for token, freq in doc_freq.items():
            self.idf[token] = __import__("math").log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def score(self, query: str, document: str) -> float:
        """ドキュメントのBM25スコアを計算。"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_len = len(doc_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in self.idf:
                continue
            freq = doc_tokens.count(token)
            idf = self.idf[token]
            numerator = freq * (self.k1 + 1)
            denominator = (
                freq + self.k1 * (1 - self.b + self.b * (doc_len / max(1, self.avgdl)))
            )
            score += idf * (numerator / denominator)

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """テキストをトークン化。"""
        text = text.lower()
        tokens = re.findall(r"[\w\d]+", text)
        return tokens


# ============================================================================
# リトライロジック付き API 呼び出し
# ============================================================================

def get_vector_store_with_retry():
    """リトライ付き vector_store 取得（最大3回）。"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            from api.knowledge.vector_store import get_store
            return get_store()
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"Failed to get vector store after {max_retries} attempts: {e}")
                raise
            import time
            wait_time = 2 ** (attempt - 1)  # 1秒, 2秒, 4秒
            logger.debug(f"Vector store retry {attempt}/{max_retries}, waiting {wait_time}s...")
            time.sleep(wait_time)


# ============================================================================
# 業種・スコア範囲フィルタ
# ============================================================================

_INDUSTRY_CATEGORIES = {
    "c": "c 製造業",
    "d": "d 建設業",
    "e": "e 電気・ガス業",
    "f": "f 水道・廃棄物業",
    "g": "g 情報通信業",
    "h": "h 運輸業",
    "i": "i 卸売業",
    "j": "j 小売業",
    "k": "k 金融・保険業",
    "l": "l 不動産業",
    "m": "m 宿泊業",
    "n": "n 飲食サービス業",
    "p": "p 医療・福祉",
    "r": "r サービス業",
}


def filter_by_industry(notes: list[dict[str, Any]], industry_code: str) -> list[dict[str, Any]]:
    """業種コード（例：'c'）でノートをフィルタ。"""
    if not industry_code or industry_code not in _INDUSTRY_CATEGORIES:
        return notes
    target_category = _INDUSTRY_CATEGORIES[industry_code]
    return [
        n for n in notes
        if not n.get("metadata", {}).get("industry")
        or target_category in n["metadata"]["industry"]
    ]


def filter_by_score_range(
    notes: list[dict[str, Any]], min_score: float, max_score: float
) -> list[dict[str, Any]]:
    """スコア範囲で過去案件ノートをフィルタ。"""
    filtered = []
    for note in notes:
        meta = note.get("metadata", {})
        score_range = meta.get("score_range")
        if not score_range:
            filtered.append(note)
            continue
        note_min, note_max = score_range
        # オーバーラップチェック
        if note_min <= max_score and note_max >= min_score:
            filtered.append(note)
    return filtered


# ============================================================================
# Wikilink トラバーサル（リンク先ノートのプリフェッチ）
# ============================================================================

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def extract_wikilinks(text: str, vault: Path) -> list[Path]:
    """テキストから Wikilink を抽出し、対応するファイルパスを返す。"""
    links = []
    for match in _WIKILINK_RE.finditer(text):
        link_name = match.group(1).strip()
        # 相対パスとして解釈（.md を追加）
        if not link_name.lower().endswith(".md"):
            link_name += ".md"
        potential_path = vault / link_name
        if potential_path.exists():
            links.append(potential_path)
    return links


def prefetch_wikilinks(
    path: Path, vault: Path, max_depth: int = 1
) -> dict[str, str]:
    """リンク先ノートを再帰的にプリフェッチ。

    Returns:
        {
            "[[link1]]": "コンテンツ（先頭500文字）",
            "[[link2]]": "...",
        }
    """
    if max_depth <= 0:
        return {}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    linked_content: dict[str, str] = {}
    for linked_path in extract_wikilinks(text, vault):
        try:
            linked_text = linked_path.read_text(encoding="utf-8", errors="ignore")
            rel = str(linked_path.relative_to(vault))
            linked_content[f"[[{rel}]]"] = linked_text[:500]
        except OSError:
            continue

    return linked_content


# ============================================================================
# 統合インデックス（メタデータ付き）
# ============================================================================

class ObsidianIndexWithMetadata:
    """Frontmatter メタデータを含むインデックス。"""

    def __init__(self):
        self.notes: dict[str, dict[str, Any]] = {}  # path -> {metadata, content, ...}
        self.bm25: BM25Scorer | None = None
        self.built_at = 0.0

    def build(self, vault: Path, knowledge_paths: list[Path]) -> None:
        """インデックスを構築。"""
        self.notes.clear()
        documents = []

        for path in knowledge_paths:
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            metadata = extract_metadata(path, content)
            self.notes[str(path)] = {
                "metadata": metadata,
                "content": content,
                "content_lower": content.lower(),
                "word_count": len(content.split()),
            }
            documents.append(content)

        # BM25 学習
        if documents:
            self.bm25 = BM25Scorer()
            self.bm25.fit(documents)

        self.built_at = time.time()

    def search_with_metadata(
        self, query: str, limit: int = 4
    ) -> list[dict[str, Any]]:
        """メタデータ + BM25 スコアで検索。"""
        query_lower = query.lower()
        results = []

        for path_str, note_data in self.notes.items():
            meta = note_data["metadata"]
            content = note_data["content"]

            # キーワードマッチ
            if query_lower not in note_data["content_lower"]:
                continue

            # BM25 スコア
            bm25_score = 0.0
            if self.bm25:
                bm25_score = self.bm25.score(query, content)

            # メタデータボーナス
            bonus = 0
            if query_lower in meta["title"].lower():
                bonus += 3.0
            if any(query_lower in tag.lower() for tag in meta.get("tags", [])):
                bonus += 2.0

            total_score = bm25_score + bonus

            results.append({
                "path": meta["path"],
                "title": meta["title"],
                "score": total_score,
                "metadata": meta,
                "snippet": content[:700],
            })

        # スコアでソート
        results.sort(key=lambda x: -x["score"])
        return results[:limit]
