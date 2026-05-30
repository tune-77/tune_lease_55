"""
PHASE 2 改善検証テスト

改善 A: キャッシュ最適化（クエリ正規化）
改善 B: 差分ドキュメント同期
"""

import logging
import tempfile
import json
import os
from pathlib import Path

# テストロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


def test_query_normalization():
    """改善 A: クエリ正規化テスト"""
    from mobile_app.rag_cache_layer import LRURAGCache

    print("\n" + "=" * 70)
    print("🧪 テスト 1: クエリ正規化（改善 A）")
    print("=" * 70)

    cache = LRURAGCache(maxsize=256)

    # テスト用クエリ
    test_queries = [
        ("飲食業リース", "飲食業 リース"),  # スペース異なり
        ("飲食業のリース", "飲食業リース"),  # 助詞あり/なし
        ("飲食業　リース", "飲食業リース"),  # 全角スペース
        ("飲食業リース", "飲食業リース"),  # 完全一致
    ]

    print("\n📌 正規化テスト:")
    for q1, q2 in test_queries:
        norm1 = cache._normalize_query(q1)
        norm2 = cache._normalize_query(q2)
        match = norm1 == norm2
        status = "✅" if match else "❌"
        print(f"{status} '{q1}' → '{norm1}'")
        print(f"   '{q2}' → '{norm2}'")
        print(f"   一致: {match}\n")

    # キャッシュヒット確認
    print("\n📌 キャッシュヒット確認:")
    result1 = {"content": "飲食業リースの基準"}
    # セット時から正規化されたキーを使う
    base_key = f"search:{cache._normalize_query('飲食業リース')}:hybrid"
    cache.set(base_key, result1)
    print(f"📝 セット時のキー: {base_key}")

    # 異なるバリエーションでアクセス
    variants = [
        "飲食業のリース",
        "飲食業 リース",
        "飲食業　リース",
    ]

    for variant in variants:
        normalized_key = f"search:{cache._normalize_query(variant)}:hybrid"
        cached = cache.get(normalized_key)
        status = "✅ HIT" if cached else "❌ MISS"
        print(f"{status}: '{variant}' → {normalized_key}")

    stats = cache.get_stats()
    print(f"\n📊 キャッシュ統計:")
    print(f"   ヒット数: {stats['hits']}")
    print(f"   ミス数: {stats['misses']}")
    print(f"   ヒット率: {stats['hit_rate']}")

    return stats['hits'] > 0  # ヒットがあれば成功


def test_differential_sync():
    """改善 B: 差分ドキュメント同期テスト"""
    import time
    from mobile_app.document_sync_tracker import DocumentSyncTracker

    print("\n" + "=" * 70)
    print("🧪 テスト 2: 差分ドキュメント同期（改善 B）")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # テスト用ドキュメントファイルを作成
        test_docs_dir = Path(tmpdir) / "docs"
        test_docs_dir.mkdir()

        doc1_path = test_docs_dir / "doc1.md"
        doc1_path.write_text("content1")

        doc2_path = test_docs_dir / "doc2.md"
        doc2_path.write_text("content2")

        # テスト用 .sync_state.json パスをオーバーライド
        original_state_file = DocumentSyncTracker.STATE_FILE
        DocumentSyncTracker.STATE_FILE = str(Path(tmpdir) / ".sync_state.json")

        try:
            tracker = DocumentSyncTracker()

            # 初期ドキュメント
            all_docs = [
                {"full_path": str(doc1_path), "title": "Doc 1"},
                {"full_path": str(doc2_path), "title": "Doc 2"},
            ]

            print("\n📌 初回同期（全件）:")
            changed, deleted = tracker.get_changed_documents(all_docs)
            print(f"✅ 変更検出: {len(changed)} 件")
            print(f"✅ 削除検出: {len(deleted)} 件")
            assert len(changed) == 2, "初回は全件変更として検出されるべき"
            assert len(deleted) == 0, "削除はないはず"

            tracker.save_state(all_docs)

            # 2 回目同期（変更なし）
            print("\n📌 2 回目同期（変更なし）:")
            # 新しいトラッカーインスタンスで状態を読み込む（STATE_FILE は既に設定済み）
            tracker2 = DocumentSyncTracker()
            changed, deleted = tracker2.get_changed_documents(all_docs)
            print(f"✅ 変更検出: {len(changed)} 件")
            print(f"✅ 削除検出: {len(deleted)} 件")
            print(f"📝 デバッグ: トラッカー状態 = {tracker2.state}")
            assert len(changed) == 0, f"変更なしのはず（実際: {len(changed)}）"
            assert len(deleted) == 0, f"削除なしのはず（実際: {len(deleted)}）"

            # 3 回目同期（ファイル更新）
            print("\n📌 3 回目同期（ファイル更新）:")
            time.sleep(1.1)  # mtime が更新されるよう待つ
            doc1_path.write_text("updated content1")  # ファイル更新
            tracker3 = DocumentSyncTracker()
            changed, deleted = tracker3.get_changed_documents(all_docs)
            print(f"✅ 変更検出: {len(changed)} 件")
            print(f"✅ 削除検出: {len(deleted)} 件")
            assert len(changed) == 1, f"1 ファイルが更新されたはず（検出: {len(changed)}）"

            tracker3.save_state(all_docs)

            # 4 回目同期（ファイル削除）
            print("\n📌 4 回目同期（ファイル削除）:")
            doc1_path.unlink()  # ファイル削除
            all_docs_after_delete = [
                {"full_path": str(doc2_path), "title": "Doc 2"},
            ]
            tracker4 = DocumentSyncTracker()
            changed, deleted = tracker4.get_changed_documents(all_docs_after_delete)
            print(f"✅ 変更検出: {len(changed)} 件")
            print(f"✅ 削除検出: {len(deleted)} 件")
            assert len(deleted) == 1, f"1 ファイルが削除されたはず（検出: {len(deleted)}）"

            return True

        finally:
            DocumentSyncTracker.STATE_FILE = original_state_file


def test_thread_safety():
    """スレッドセーフ確認テスト"""
    import threading
    from mobile_app.rag_cache_layer import LRURAGCache

    print("\n" + "=" * 70)
    print("🧪 テスト 3: スレッドセーフティ確認")
    print("=" * 70)

    cache = LRURAGCache(maxsize=256)
    results = {"errors": 0}

    def worker(thread_id, operations=100):
        """複数スレッドでキャッシュ操作"""
        try:
            for i in range(operations):
                key = f"key_{thread_id}_{i}"
                cache.set(key, f"value_{i}")

                # 同期アクセス
                value = cache.get(key)
                if value != f"value_{i}":
                    results["errors"] += 1
        except Exception as e:
            logger.error(f"❌ Thread {thread_id} エラー: {e}")
            results["errors"] += 1

    print("\n📌 マルチスレッドアクセス（5 スレッド × 100 操作）:")
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if results["errors"] == 0:
        print("✅ エラーなし - スレッドセーフ")
        return True
    else:
        print(f"❌ {results['errors']} 個のエラー発生")
        return False


def run_all_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("🚀 PHASE 2 改善検証テスト開始")
    print("=" * 70)

    tests = [
        ("クエリ正規化", test_query_normalization),
        ("差分同期", test_differential_sync),
        ("スレッドセーフティ", test_thread_safety),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} テスト失敗: {e}")
            results[test_name] = False

    # 結果サマリ
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリ")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n🎯 結果: {passed}/{total} テスト合格")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
