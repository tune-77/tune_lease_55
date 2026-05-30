"""
RAG キャッシング層（LRU キャッシュ + 高速化）

フェーズ3: キャッシング・最適化
- LRU キャッシュ導入
- クエリ結果キャッシング
- レイテンシ < 100ms 達成
"""

import time
import logging
import re
import threading
from functools import lru_cache, wraps
from typing import Any, Callable
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LRURAGCache:
    """LRU キャッシュ実装"""

    def __init__(self, maxsize: int = 256, ttl_seconds: int = 3600):
        """
        初期化

        Args:
            maxsize: キャッシュサイズ
            ttl_seconds: キャッシュ有効期限（秒）
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # PHASE 2: スレッドセーフ対応

    @staticmethod
    def _normalize_query(query: str) -> str:
        """
        PHASE 2: クエリを正規化してキャッシュヒット率を上げる

        同義のクエリ（例: "飲食業リース" と "飲食業のリース"）を同じキーで扱う

        Args:
            query: 元のクエリ文字列

        Returns:
            正規化されたクエリ
        """
        q = query.strip().lower()
        q = re.sub(r'[\s　]+', '', q)  # 全角・半角スペースを削除
        q = re.sub(r'[のがをはもで]', '', q)  # 日本語助詞を除去（最小限）
        return q

    def get(self, key: str) -> Any | None:
        """キャッシュから取得（PHASE 2: スレッドセーフ対応）"""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp = self.cache[key]

            # TTL チェック
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                self.misses += 1
                return None

            # LRU: 最近使用したアイテムを最後に移動
            self.cache.move_to_end(key)
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any):
        """キャッシュに設定（PHASE 2: スレッドセーフ対応）"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            self.cache[key] = (value, datetime.now())

            # 容量超過時は最古のアイテムを削除
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    def clear(self):
        """キャッシュをクリア"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl_seconds
        }


class RAGCacheDecorator:
    """RAG キャッシング デコレータ"""
    
    def __init__(self, cache: LRURAGCache):
        self.cache = cache
    
    def cached(self, func: Callable) -> Callable:
        """関数をキャッシュでラップ"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キャッシュキーを生成（関数名 + 引数）
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # キャッシュ取得
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"💾 キャッシュヒット: {cache_key[:50]}")
                return cached_result
            
            # キャッシュミス → 関数実行
            logger.debug(f"🔄 キャッシュミス: {cache_key[:50]}")
            result = func(*args, **kwargs)
            
            # キャッシュに保存
            self.cache.set(cache_key, result)
            return result
        
        return wrapper


class OptimizedRAGPipeline:
    """最適化された RAG パイプライン"""
    
    def __init__(self):
        self.cache = LRURAGCache(maxsize=512, ttl_seconds=3600)
        self.cache_decorator = RAGCacheDecorator(self.cache)
        self.query_times = []
    
    def retrieve_cached(self, query: str, retriever: Any) -> tuple[list[dict], float]:
        """
        キャッシュ付きで検索

        Args:
            query: 検索クエリ
            retriever: RAG retriever

        Returns:
            (結果, レイテンシ)
        """
        start_time = time.time()

        # PHASE 2: クエリ正規化でキャッシュヒット率を向上
        normalized_query = self.cache._normalize_query(query)
        cache_key = f"retrieve:{normalized_query}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            elapsed = time.time() - start_time
            logger.info(f"💾 キャッシュから取得: {elapsed*1000:.2f}ms")
            return cached_result, elapsed
        
        # 実検索
        results = retriever.retrieve(query, top_k=5)
        elapsed = time.time() - start_time
        
        # キャッシュに保存
        self.cache.set(cache_key, results)
        
        logger.info(f"🔍 検索完了: {elapsed*1000:.2f}ms")
        self.query_times.append(elapsed)
        
        return results, elapsed
    
    def get_performance_metrics(self) -> dict:
        """パフォーマンス指標を取得"""
        if not self.query_times:
            return {}
        
        import statistics
        
        return {
            "cache_stats": self.cache.get_stats(),
            "avg_latency_ms": statistics.mean(self.query_times) * 1000,
            "median_latency_ms": statistics.median(self.query_times) * 1000,
            "min_latency_ms": min(self.query_times) * 1000,
            "max_latency_ms": max(self.query_times) * 1000,
            "p95_latency_ms": (sorted(self.query_times)[int(len(self.query_times) * 0.95)] * 1000 
                               if len(self.query_times) > 1 else 0)
        }


# テスト用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 RAG キャッシング層 テスト")
    print("=" * 60)
    
    cache = LRURAGCache(maxsize=10, ttl_seconds=60)
    
    # テスト: キャッシュ設定・取得
    cache.set("query1", {"result": "test1"})
    cache.set("query2", {"result": "test2"})
    
    result = cache.get("query1")
    print(f"✅ 取得: {result}")
    
    # 統計
    cache.get("query1")  # ヒット
    cache.get("query3")  # ミス
    
    stats = cache.get_stats()
    print("\n📊 キャッシュ統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
