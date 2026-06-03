"""
BM25 フルテキスト検索エンジン（日本語最適化）

BM25 アルゴリズムを使用した高速フルテキスト検索
- 日本語テキストの自動トークン化（形態素解析）
- TF-IDF ベースのスコアリング
- セマンティック検索との組み合わせで Hybrid Search を実現
"""

import logging
from typing import List, Dict, Tuple, Optional
import time

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    logging.warning("⚠️  rank_bm25 not installed. Install with: pip install rank_bm25")

try:
    import fugashi
    import ipadic
    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False
    logging.warning("⚠️  fugashi not installed. Install with: pip install fugashi ipadic")

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """
    BM25 フルテキスト検索エンジン（日本語最適化）

    属性:
        bm25: BM25Okapi インスタンス
        documents: インデックス済みドキュメント
        tokenizer: 日本語形態素解析器（fugashi）
    """

    def __init__(self):
        """初期化"""
        if not RANK_BM25_AVAILABLE:
            raise ImportError("rank_bm25 is required. Install with: pip install rank_bm25")

        if not FUGASHI_AVAILABLE:
            logger.warning("⚠️  fugashi not available. Using simplified tokenization.")

        self.bm25 = None
        self.documents = []
        self.tokenizer = fugashi.GenericTagger('-d ' + ipadic.DICDIR) if FUGASHI_AVAILABLE else None

        logger.info("✅ BM25 検索エンジン初期化完了")

    def _tokenize_ja(self, text: str) -> List[str]:
        """
        日本語テキストをトークン化

        引数:
            text: 日本語テキスト

        返り値:
            トークンのリスト
        """
        if not FUGASHI_AVAILABLE or self.tokenizer is None:
            # フォールバック: 文字を 2-3 文字ずつ分割
            return [text[i:i+3] for i in range(0, len(text), 3)]

        try:
            words = self.tokenizer(text)
            # スペースと記号を除外
            return [
                word.surface
                for word in words
                if word.char_type not in ('SPACE', 'SYMBOL')
            ]
        except Exception as e:
            logger.warning(f"⚠️  Tokenization error: {e}. Using fallback.")
            return [text[i:i+3] for i in range(0, len(text), 3)]

    def index_documents(self, documents: List[Dict]):
        """
        ドキュメントをインデックス

        引数:
            documents: ドキュメントのリスト
                [
                    {'id': '1', 'title': '...', 'content': '...'},
                    ...
                ]
        """
        if not documents:
            logger.warning("⚠️  No documents to index")
            return

        logger.info(f"🔄 BM25: {len(documents)} ドキュメントをインデックス中...")
        start_time = time.time()

        self.documents = documents

        # ドキュメントをトークン化
        tokenized_corpus = []
        for doc in documents:
            content = doc.get('content', '') or ''
            title = doc.get('title', '') or ''
            # タイトルと内容を結合（タイトルは 2 倍重視）
            full_text = f"{title} {title} {content}"
            tokens = self._tokenize_ja(full_text)
            tokenized_corpus.append(tokens)

        # BM25 インデックスを構築
        self.bm25 = BM25Okapi(tokenized_corpus)

        elapsed = time.time() - start_time
        logger.info(f"✅ BM25 インデックス完了 ({elapsed:.2f}s, {len(documents)} docs)")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        クエリに対して BM25 検索を実行

        引数:
            query: 検索クエリ
            top_k: 返す結果数

        返り値:
            (ドキュメント, スコア) のタプルのリスト
        """
        if self.bm25 is None:
            logger.warning("⚠️  BM25: インデックスが初期化されていません")
            return []

        if not query or not query.strip():
            logger.warning("⚠️  BM25: 空のクエリです")
            return []

        start_time = time.time()

        # クエリをトークン化
        tokenized_query = self._tokenize_ja(query)

        # BM25 スコアを計算
        scores = self.bm25.get_scores(tokenized_query)

        # スコアと ドキュメントをペアリング
        ranked = [
            (self.documents[i], float(scores[i]))
            for i in range(len(self.documents))
            if scores[i] > 0  # スコアが 0 以上のみ
        ]

        # スコアでソート
        ranked.sort(key=lambda x: x[1], reverse=True)

        # 上位 top_k を返却
        result = ranked[:top_k]

        elapsed = time.time() - start_time
        logger.debug(f"🔍 BM25 検索完了 ({elapsed*1000:.2f}ms, {len(result)} results)")

        return result

    def get_stats(self) -> Dict:
        """インデックス統計を取得"""
        return {
            'total_documents': len(self.documents),
            'indexed': self.bm25 is not None,
            'tokenizer_available': FUGASHI_AVAILABLE
        }


class SimpleBM25SearchEngine:
    """
    シンプル BM25 検索エンジン（非依存版）

    rank_bm25 がインストールされていない場合の代替実装
    TF-IDF ベースの簡易検索を提供
    """

    def __init__(self):
        """初期化"""
        self.documents = []
        self.term_index = {}  # {term: [doc_id, ...]}
        self.doc_term_count = {}  # {doc_id: {term: count, ...}}

        logger.info("✅ Simple BM25 検索エンジン初期化完了（非依存版）")

    def _tokenize_simple(self, text: str) -> List[str]:
        """簡易トークン化（形態素解析なし）"""
        import re
        # 日本語と英数字を保持
        text = text.lower()
        # 連続した空白を単一スペースに
        text = re.sub(r'\s+', ' ', text)
        # 単語に分割
        return text.split()

    def index_documents(self, documents: List[Dict]):
        """ドキュメントをインデックス"""
        logger.info(f"🔄 Simple BM25: {len(documents)} ドキュメントをインデックス中...")

        self.documents = documents
        self.term_index = {}
        self.doc_term_count = {}

        for doc_id, doc in enumerate(documents):
            content = doc.get('content', '') or ''
            title = doc.get('title', '') or ''
            full_text = f"{title} {title} {content}"

            tokens = self._tokenize_simple(full_text)

            # term_index を更新
            self.doc_term_count[doc_id] = {}
            for token in tokens:
                if token not in self.term_index:
                    self.term_index[token] = []
                if doc_id not in self.term_index[token]:
                    self.term_index[token].append(doc_id)

                # ドキュメント内のトークン出現回数
                self.doc_term_count[doc_id][token] = self.doc_term_count[doc_id].get(token, 0) + 1

        logger.info(f"✅ Simple BM25 インデックス完了 ({len(documents)} docs)")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """シンプル BM25 検索を実行"""
        if not self.documents:
            return []

        tokens = self._tokenize_simple(query)

        # 各ドキュメントのスコアを計算（TF-IDF ベース）
        scores = {}
        for doc_id in range(len(self.documents)):
            score = 0.0
            for token in tokens:
                if token in self.term_index:
                    # TF (Token Frequency)
                    tf = self.doc_term_count[doc_id].get(token, 0)
                    # IDF (Inverse Document Frequency)
                    idf = len(self.documents) / len(self.term_index[token])
                    score += tf * idf

            if score > 0:
                scores[doc_id] = score

        # スコアでソート
        ranked = [
            (self.documents[doc_id], score)
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return ranked[:top_k]


def get_bm25_engine() -> BM25SearchEngine:
    """BM25 検索エンジンを取得（自動フォールバック）"""
    try:
        return BM25SearchEngine()
    except ImportError:
        logger.warning("⚠️  Falling back to SimpleBM25SearchEngine")
        return SimpleBM25SearchEngine()
