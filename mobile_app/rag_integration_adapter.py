"""
RAG 統合アダプタ - 既存の chat_assistant.py に RAG システムを統合

レガシーインターフェース:
  - collect_obsidian_context(query, limit=4) → ハイブリッド検索結果

新RAGシステムをレガシーAPI互換で提供
"""

import logging
from typing import Optional, Callable
from mobile_app.integrated_rag_pipeline import IntegratedRAGSystem

logger = logging.getLogger(__name__)

# グローバルRAGインスタンス
_rag_system: Optional[IntegratedRAGSystem] = None


def get_rag_system() -> IntegratedRAGSystem:
    """グローバル RAG システムインスタンスを取得（遅延初期化）"""
    global _rag_system
    if _rag_system is None:
        try:
            logger.info("🚀 RAG システムを初期化中...")
            _rag_system = IntegratedRAGSystem()
            logger.info("✅ RAG システム初期化完了")
        except Exception as e:
            logger.warning(f"⚠️ RAG システム初期化失敗: {e}")
            _rag_system = None
    return _rag_system


def collect_obsidian_context_with_rag(
    query: str, limit: int = 4
) -> list[dict[str, str]]:
    """
    RAG システムを使用した Obsidian コンテキスト収集
    
    既存の collect_obsidian_context() との互換性を保つ
    
    Args:
        query: 検索クエリ
        limit: 結果数上限
        
    Returns:
        [{"path": "ファイルパス", "snippet": "スニペット", "similarity": スコア}, ...]
    """
    rag_system = get_rag_system()
    
    if rag_system is None:
        # RAG システムが初期化失敗時は、従来の検索にフォールバック
        logger.info("🔄 フォールバック: 従来の obsidian_bridge を使用")
        from obsidian_bridge import collect_obsidian_context as legacy_collect
        return legacy_collect(query, limit=limit)
    
    # ハイブリッド検索実行
    try:
        result = rag_system.search(query, search_type="hybrid")
        
        # 結果をレガシーフォーマットに変換
        hits = []
        for item in result.get("results", [])[:limit]:
            # ベクトルDB結果またはセマンティック検索結果
            hits.append({
                "path": item.get("path", item.get("title", "unknown")),
                "snippet": item.get("content", "")[:300],  # 最初の300文字
                "similarity": item.get(
                    "combined_score",
                    item.get("similarity_score", 0)
                ),
                "method": result.get("method", "hybrid")
            })
        
        logger.info(f"✅ RAG 検索完了: {len(hits)} 件 ({result['latency_ms']:.2f}ms)")
        return hits
    
    except Exception as e:
        logger.error(f"❌ RAG 検索エラー: {e}")
        # エラー時はフォールバック
        from obsidian_bridge import collect_obsidian_context as legacy_collect
        return legacy_collect(query, limit=limit)


def get_rag_performance_metrics() -> dict:
    """RAG パフォーマンスメトリクスを取得"""
    rag_system = get_rag_system()
    
    if rag_system is None:
        return {
            "status": "RAG システム未初期化",
            "metrics": {}
        }
    
    return {
        "status": "RAG システム稼働中",
        "metrics": rag_system.get_performance_report()
    }


# ===== レガシー互換ラッパー

def collect_obsidian_context_hybrid(
    query: str,
    legacy_fn: Optional[Callable] = None,
    limit: int = 4
) -> list[dict[str, str]]:
    """
    ハイブリッド検索ラッパー
    
    新 RAG システムが利用可能なら使用、失敗時はレガシー関数にフォールバック
    
    Args:
        query: 検索クエリ
        legacy_fn: フォールバック用レガシー関数
        limit: 結果数上限
    """
    return collect_obsidian_context_with_rag(query, limit=limit)


# ===== テスト用

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print("🎯 RAG 統合アダプタ テスト")
    print("=" * 70 + "\n")
    
    # テストクエリ
    test_queries = [
        "飲食業のリース審査",
        "自己資本比率",
        "条件付き承認",
    ]
    
    for query in test_queries:
        results = collect_obsidian_context_with_rag(query, limit=3)
        print(f"Q: {query}")
        print(f"  結果数: {len(results)}")
        for i, hit in enumerate(results, 1):
            print(f"  {i}. {hit['path'][:50]}")
            print(f"     スコア: {hit.get('similarity', 0):.3f}")
        print()
    
    # パフォーマンスレポート
    metrics = get_rag_performance_metrics()
    print(f"\n📊 {metrics['status']}")
    if metrics['metrics']:
        report = metrics['metrics']
        print(f"  キャッシュ統計: {report.get('cache_stats', {})}")
        print(f"  レイテンシ目標: < 100ms")
        if 'latency' in report:
            print(f"  実測 p95: {report['latency'].get('p95_ms', 'N/A'):.2f}ms")
    
    print("\n" + "=" * 70 + "\n")
