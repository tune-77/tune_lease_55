"""
統合 RAG パイプライン（セマンティック検索 + ベクトル DB + キャッシング）

高性能リース審査 AI 検索システム
- セマンティック検索（embedding + cosine similarity）
- ベクトル DB（高速検索）
- キャッシング層（< 100ms レイテンシ）
"""

import logging
import time
from typing import Any, Optional
from mobile_app.semantic_search_engine import SemanticRAGRetriever, SemanticSearchEngine
from mobile_app.vector_db import LocalVectorDB
from mobile_app.rag_cache_layer import OptimizedRAGPipeline

logger = logging.getLogger(__name__)


class IntegratedRAGSystem:
    """統合 RAG システム"""
    
    def __init__(self):
        """初期化"""
        logger.info("🚀 統合 RAG システムの初期化中...")
        
        # コンポーネント初期化
        self.retriever = SemanticRAGRetriever()
        self.vector_db = LocalVectorDB(db_name="lease_rag")
        self.pipeline = OptimizedRAGPipeline()
        
        # ドキュメント同期
        self._sync_documents()
        
        logger.info("✅ RAG システム初期化完了")
    
    def _sync_documents(self):
        """Obsidian ドキュメントをベクトル DB に同期"""
        logger.info("🔄 ドキュメント同期中...")
        
        docs_to_sync = [
            {
                "id": doc["path"],
                "title": doc["title"],
                "path": doc["path"],
                "content": doc["content"]
            }
            for doc in self.retriever.obsidian_documents
        ]
        
        self.vector_db.upsert(docs_to_sync)
        logger.info(f"✅ {len(docs_to_sync)} 個のドキュメントを同期")
    
    def search(self, query: str, search_type: str = "hybrid") -> dict:
        """
        ハイブリッド検索（キャッシュ + セマンティック + ベクトル DB）
        
        Args:
            query: 検索クエリ
            search_type: "hybrid" / "semantic" / "vector"
            
        Returns:
            {
                "query": クエリ,
                "results": [結果リスト],
                "latency_ms": レイテンシ,
                "method": 使用方法,
                "cache_hit": キャッシュヒット有無
            }
        """
        start_time = time.time()
        
        # キャッシュ確認
        cache_key = f"search:{query}:{search_type}"
        cached = self.pipeline.cache.get(cache_key)
        
        if cached is not None:
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"💾 キャッシュヒット: {elapsed:.2f}ms")
            return {
                **cached,
                "latency_ms": elapsed,
                "cache_hit": True
            }
        
        # 検索実行
        results = []
        method = "unknown"
        
        if search_type == "hybrid":
            # ハイブリッド検索（セマンティック + ベクトル DB）
            semantic_results = self.retriever.retrieve(query, top_k=3)
            vector_results = self.vector_db.query(query, top_k=3)
            
            # マージしてスコアでソート
            merged = {}
            for r in semantic_results:
                merged[r.get("path")] = {
                    **r,
                    "semantic_score": r.get("similarity_score", 0),
                    "vector_score": 0
                }
            
            for r in vector_results:
                if r.get("path") in merged:
                    merged[r["path"]]["vector_score"] = r["similarity_score"]
                else:
                    merged[r.get("path")] = {
                        **r,
                        "semantic_score": 0,
                        "vector_score": r["similarity_score"]
                    }
            
            # 複合スコア計算
            results = []
            for path, item in merged.items():
                combined_score = (item["semantic_score"] + item["vector_score"]) / 2
                item["combined_score"] = combined_score
                results.append(item)
            
            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            method = "hybrid"
        
        elif search_type == "semantic":
            results = self.retriever.retrieve(query, top_k=5)
            method = "semantic"
        
        elif search_type == "vector":
            results = self.vector_db.query(query, top_k=5)
            method = "vector"
        
        elapsed = (time.time() - start_time) * 1000
        
        # 結果をキャッシュに保存
        response = {
            "query": query,
            "results": results[:5],  # top 5
            "latency_ms": elapsed,
            "method": method,
            "cache_hit": False
        }
        
        self.pipeline.cache.set(cache_key, response)
        
        logger.info(f"✅ 検索完了: {method} ({elapsed:.2f}ms)")
        return response
    
    def get_performance_report(self) -> dict:
        """パフォーマンスレポート（フェーズ3: レイテンシ最適化）"""
        metrics = self.pipeline.get_performance_metrics()
        
        return {
            "title": "RAG システム パフォーマンスレポート",
            "cache_stats": metrics.get("cache_stats", {}),
            "latency": {
                "avg_ms": metrics.get("avg_latency_ms", 0),
                "median_ms": metrics.get("median_latency_ms", 0),
                "p95_ms": metrics.get("p95_latency_ms", 0),
                "min_ms": metrics.get("min_latency_ms", 0),
                "max_ms": metrics.get("max_latency_ms", 0),
                "target_ms": 100,
                "status": "✅ 目標達成" if metrics.get("p95_latency_ms", 999) < 100 else "🔄 改善中"
            },
            "vector_db": self.vector_db.get_stats(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


# テスト用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print("🎯 統合 RAG システムテスト（フェーズ1-3 統合）")
    print("=" * 70)
    
    rag_system = IntegratedRAGSystem()
    
    # テストクエリ
    test_queries = [
        "飲食業のリース審査",
        "自己資本比率30%",
        "条件付き承認の対策",
    ]
    
    print("\n📌 ハイブリッド検索テスト:\n")
    for query in test_queries:
        result = rag_system.search(query, search_type="hybrid")
        print(f"  Q: {query}")
        print(f"  ├─ 方法: {result['method']}")
        print(f"  ├─ レイテンシ: {result['latency_ms']:.2f}ms")
        print(f"  └─ 結果数: {len(result['results'])}")
    
    # 2回目のクエリ（キャッシュヒット）
    print(f"\n💾 キャッシュテスト:\n")
    result = rag_system.search("飲食業のリース審査", search_type="hybrid")
    print(f"  Q: 飲食業のリース審査 (2回目)")
    print(f"  ├─ キャッシュヒット: {result['cache_hit']}")
    print(f"  └─ レイテンシ: {result['latency_ms']:.2f}ms")
    
    # パフォーマンスレポート
    print("\n📊 パフォーマンスレポート:\n")
    report = rag_system.get_performance_report()
    print(f"  キャッシュ統計:")
    for k, v in report["cache_stats"].items():
        print(f"    {k}: {v}")
    print(f"\n  レイテンシ:")
    for k, v in report["latency"].items():
        print(f"    {k}: {v}")
    
    print("\n" + "=" * 70)
    print("✅ 統合テスト完了 - RAG システム稼働中")
    print("=" * 70 + "\n")
