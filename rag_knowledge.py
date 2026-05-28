"""
RAGナレッジ検索エンジン。
knowledge_base.json・leasing_knowhow.json を TF-IDF ベクトル化し、
クエリに最も関連するチャンクを返す。外部ベクトルDBは不要（sklearn のみ使用）。
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE_FILE = os.path.join(BASE_DIR, "static_data", "knowledge_base.json")
_KNOWHOW_FILE = os.path.join(BASE_DIR, "static_data", "leasing_knowhow.json")

# モジュールレベルのシングルトン（初回呼び出し時に構築）
_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None
_chunks: list["Chunk"] = []


class Chunk(NamedTuple):
    text: str
    source: str  # "faq" / "manual" / "case" / "industry" / "knowhow" 等
    title: str   # 検索結果表示用の見出し


def _load_json(path: str) -> dict | list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _split_text(text: str, max_len: int = 280) -> list[str]:
    """長いテキストを句読点単位で max_len 以下に分割する。"""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    buf = ""
    for char in text:
        buf += char
        if char in ("。", "．", "\n", "?", "？") and len(buf) >= 60:
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts or [text[:max_len]]


def _extract_chunks() -> list[Chunk]:
    """knowledge_base.json と leasing_knowhow.json からチャンクを生成する。"""
    chunks: list[Chunk] = []
    kb = _load_json(_KNOWLEDGE_FILE)

    # ─── FAQ ───────────────────────────────────────────────────────────────
    for faq in (kb.get("faq") or []):
        q = (faq.get("q") or "").strip()
        a = (faq.get("a") or "").strip()
        cat = faq.get("category", "")
        if q and a:
            text = f"Q: {q}\nA: {a}"
            for part in _split_text(text):
                chunks.append(Chunk(text=part, source="faq", title=f"FAQ/{cat}: {q[:40]}"))

    # ─── 審査マニュアル ────────────────────────────────────────────────────
    manual = kb.get("manual") or {}
    pre = manual.get("pre_screening") or {}
    for item in pre.get("required_items") or []:
        chunks.append(Chunk(text=f"審査前必要書類: {item}", source="manual", title="審査前チェックリスト"))
    for cond in pre.get("auto_reject_conditions") or []:
        chunks.append(Chunk(text=f"自動否決条件: {cond}", source="manual", title="自動否決条件"))
    bh = manual.get("borderline_handling") or {}
    for key, val in bh.items():
        cond = (val.get("condition") or "").strip()
        req = (val.get("required") or "").strip()
        if cond:
            text = f"ボーダーライン処理 ({key}): {cond}。必要条件: {req}"
            for part in _split_text(text):
                chunks.append(Chunk(text=part, source="manual", title=f"ボーダーライン/{key}"))

    # ─── スコアリングシステム ──────────────────────────────────────────────
    ss = kb.get("scoring_system") or {}
    overview = (ss.get("overview") or "").strip()
    if overview:
        chunks.append(Chunk(text=f"スコアリング概要: {overview}", source="scoring", title="スコアリング概要"))
    for grade in (ss.get("grade_boundaries") or []):
        text = f"ランク{grade.get('grade')}: {grade.get('label')} — {grade.get('policy')}"
        chunks.append(Chunk(text=text, source="scoring", title="ランク境界"))
    for item in (ss.get("qualitative_items") or []):
        label = item.get("label", "")
        for opt in (item.get("options") or []):
            text = f"定性評価「{label}」{opt.get('score')}点: {opt.get('label')} — {opt.get('detail','')}"
            chunks.append(Chunk(text=text, source="scoring", title=f"定性項目/{label}"))

    # ─── 業種別ガイド ──────────────────────────────────────────────────────
    for cat in (kb.get("industry_guide") or {}).get("categories") or []:
        name = cat.get("name", "")
        ki = cat.get("key_indicators") or {}
        if ki:
            parts_txt = "、".join(f"{k}: {v}" for k, v in ki.items())
            chunks.append(Chunk(
                text=f"業種「{name}」主要指標目安: {parts_txt}",
                source="industry",
                title=f"業種ガイド/{name}",
            ))
        for tip in (cat.get("scoring_tips") or []):
            chunks.append(Chunk(text=f"業種「{name}」審査ポイント: {tip}", source="industry", title=f"業種ガイド/{name}"))
        risks = cat.get("risks") or []
        if risks:
            chunks.append(Chunk(
                text=f"業種「{name}」主なリスク: {' / '.join(risks)}",
                source="industry",
                title=f"業種リスク/{name}",
            ))

    # ─── 審査事例集 ────────────────────────────────────────────────────────
    for case in (kb.get("cases") or []):
        title = case.get("title", "")
        decision = case.get("decision", "")
        lessons = case.get("lessons") or []
        ind = case.get("industry", "")
        text = f"審査事例「{title}」（{ind}）: 判定={decision}。" + " ".join(lessons[:3])
        for part in _split_text(text):
            chunks.append(Chunk(text=part, source="case", title=f"事例/{title}"))

    # ─── スコア改善ガイド ──────────────────────────────────────────────────
    guide = kb.get("improvement_guide") or {}
    for item in (guide.get("quick_wins") or []):
        text = f"クイックウィン: {item.get('action','')} → {item.get('impact','')}（{item.get('timeline','')}）"
        chunks.append(Chunk(text=text, source="improvement", title="スコア改善/クイックウィン"))
    for item in (guide.get("medium_term") or []):
        text = f"中期改善: {item.get('action','')} → {item.get('impact','')}（{item.get('timeline','')}）"
        chunks.append(Chunk(text=text, source="improvement", title="スコア改善/中期"))

    # ─── leasing_knowhow.json ──────────────────────────────────────────────
    knowhow = _load_json(_KNOWHOW_FILE)
    if isinstance(knowhow, dict):
        for section_key, section_val in knowhow.items():
            if isinstance(section_val, list):
                for entry in section_val:
                    if isinstance(entry, dict):
                        situation = (entry.get("situation") or entry.get("title") or "").strip()
                        strategy = (entry.get("strategy") or entry.get("content") or entry.get("description") or "").strip()
                        if situation or strategy:
                            text = f"ノウハウ「{situation}」: {strategy}"
                            for part in _split_text(text):
                                chunks.append(Chunk(text=part, source="knowhow", title=f"ノウハウ/{section_key}"))
            elif isinstance(section_val, str) and section_val.strip():
                for part in _split_text(f"{section_key}: {section_val}"):
                    chunks.append(Chunk(text=part, source="knowhow", title=f"ノウハウ/{section_key}"))

    # 空テキストを除去
    return [c for c in chunks if len(c.text.strip()) >= 10]


def _build_index() -> None:
    """TF-IDF インデックスを構築する（モジュール初期化時に1回だけ実行）。"""
    global _vectorizer, _tfidf_matrix, _chunks
    _chunks = _extract_chunks()
    if not _chunks:
        return
    texts = [c.text for c in _chunks]
    _vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # 日本語は文字n-gramが有効
        ngram_range=(2, 3),
        max_features=20000,
        sublinear_tf=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(texts)


class RetrievedChunk(NamedTuple):
    chunk: Chunk
    score: float


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    クエリに最も関連するチャンクを返す。

    Parameters
    ----------
    query : ユーザーの質問文字列
    top_k : 返すチャンク数（デフォルト5）

    Returns
    -------
    スコア降順の RetrievedChunk リスト
    """
    global _vectorizer, _tfidf_matrix, _chunks

    if _vectorizer is None:
        _build_index()

    if _vectorizer is None or _tfidf_matrix is None or not _chunks:
        return []

    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(sims)[::-1][:top_k]
    results: list[RetrievedChunk] = []
    for idx in top_indices:
        score = float(sims[idx])
        if score < 0.01:
            break
        results.append(RetrievedChunk(chunk=_chunks[idx], score=score))
    return results


def build_rag_context(query: str, top_k: int = 5) -> str:
    """
    クエリに関連するチャンクを取得し、プロンプト挿入用のコンテキスト文字列を生成する。
    knowledge.build_knowledge_context() の代替として使用できる。
    """
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return ""

    lines = ["=== RAGナレッジ検索結果 ==="]
    for i, hit in enumerate(hits, 1):
        lines.append(f"\n[{i}] ({hit.chunk.source}) {hit.chunk.title}")
        lines.append(hit.chunk.text)
    return "\n".join(lines)


def get_index_stats() -> dict:
    """インデックスの統計情報を返す（UI表示用）。"""
    if _vectorizer is None:
        _build_index()
    return {
        "chunk_count": len(_chunks),
        "vocab_size": len(_vectorizer.vocabulary_) if _vectorizer else 0,
        "sources": _count_sources(),
    }


def _count_sources() -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in _chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    return counts
