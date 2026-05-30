"""
統合 RAG + ChromaDB メンテナンススクリプト

【毎朝3時実行】
✅ ChromaDB 再インデックス（既存）
✅ LocalVectorDB 同期（新規）
✅ キャッシュ統計レポート
✅ パフォーマンス監視
✅ 異常検知アラート

【統合により】
✅ ファイル読み込み 1回で済む
✅ インデックス不整合なし
✅ 実行時間短縮
✅ CPU/メモリ効率化
"""

import logging
import json
import os
from datetime import datetime
from mobile_app.integrated_rag_pipeline import IntegratedRAGSystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ===== ChromaDB 再インデックス機能

def run_chroma_reindex(vault_path: str, full_mode: bool = True) -> dict:
    """
    ChromaDB 再インデックス（既存処理を統合）

    Args:
        vault_path: Obsidian Vault パス
        full_mode: True=全件再構築、False=差分更新

    Returns:
        {"added": 件数, "skipped": 件数, "status": "success/error"}
    """
    try:
        import chromadb
        from api.knowledge.vector_store import KnowledgeVectorStore, _CHROMA_DIR, _COLLECTION_NAME
        from api.knowledge.obsidian_loader import scan_vault

        logger.info(f"🔄 ChromaDB 再インデックス開始（full_mode={full_mode}）")

        if full_mode:
            # コレクション削除
            client = chromadb.PersistentClient(path=_CHROMA_DIR)
            try:
                client.delete_collection(_COLLECTION_NAME)
                logger.info("✅ 既存 ChromaDB コレクション削除完了")
            except Exception as e:
                logger.info(f"⚠️  削除対象コレクションなし（初回実行）")

            # 全件インデックス
            store = KnowledgeVectorStore()
            logger.info(f"📚 Vault スキャン開始: {vault_path}")

            pending, added, skipped = [], 0, 0
            seen_ids: set[str] = set()

            for chunk in scan_vault(vault_path):
                if chunk.doc_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(chunk.doc_id)
                pending.append(chunk)

                if len(pending) >= 50:
                    added += store.upsert_chunks(pending)
                    logger.info(f"   ... {added} 件登録済み")
                    pending.clear()

            if pending:
                added += store.upsert_chunks(pending)

            logger.info(f"✅ ChromaDB インデックス完了: {added} 件追加, {skipped} 件スキップ")

            return {
                "added": added,
                "skipped": skipped,
                "status": "success"
            }

        else:
            # 差分更新
            from api.knowledge.indexer import run_indexing
            added, skipped = run_indexing(vault_path)
            logger.info(f"✅ ChromaDB 差分更新完了: {added} 件更新, {skipped} 件スキップ")

            return {
                "added": added,
                "skipped": skipped,
                "status": "success"
            }

    except ImportError as e:
        logger.warning(f"⚠️  ChromaDB モジュール未インストール（スキップ）")
        return {
            "status": "skipped",
            "reason": "ChromaDB not installed"
        }

    except Exception as e:
        logger.error(f"❌ ChromaDB 再インデックスエラー: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# ===== 統合メンテナンス関数

def run_daily_maintenance():
    """毎朝3時に実行する統合メンテナンス"""
    
    print("\n" + "=" * 80)
    print(f"🔄 統合メンテナンス実行（RAG + ChromaDB）")
    print(f"   実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    vault_path = os.environ.get(
        "OBSIDIAN_VAULT_PATH",
        "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
    )
    
    maintenance_report = {
        "timestamp": datetime.now().isoformat(),
        "vault_path": vault_path,
        "chroma_reindex": {},
        "rag_maintenance": {},
        "alerts": [],
        "status": "success"
    }
    
    try:
        # ================== タスク0: ChromaDB 再インデックス ==================
        print("🔵 【タスク0】ChromaDB 再インデックス中...")
        chroma_result = run_chroma_reindex(vault_path, full_mode=True)
        maintenance_report["chroma_reindex"] = chroma_result
        
        if chroma_result.get("status") == "success":
            print(f"   ✅ {chroma_result.get('added', 0)} 件追加, {chroma_result.get('skipped', 0)} 件スキップ\n")
        elif chroma_result.get("status") == "skipped":
            print(f"   ⏭️  スキップ（理由: {chroma_result.get('reason', 'unknown')}）\n")
        else:
            print(f"   ❌ エラー: {chroma_result.get('error', 'unknown')}\n")
            maintenance_report["alerts"].append(f"ChromaDB 再インデックス失敗: {chroma_result.get('error')}")
        
        # ================== タスク1: LocalVectorDB 同期 ==================
        print("📚 【タスク1】LocalVectorDB ドキュメント同期中...")
        rag_system = IntegratedRAGSystem()
        rag_system._sync_documents()
        rag_doc_count = len(rag_system.retriever.obsidian_documents)
        print(f"   ✅ {rag_doc_count} 個のドキュメントを同期\n")
        
        # ================== タスク2: パフォーマンスレポート ==================
        print("📊 【タスク2】パフォーマンスレポート生成中...")
        report = rag_system.get_performance_report()
        maintenance_report["rag_maintenance"] = report
        
        cache_stats = report.get('cache_stats', {})
        print(f"   キャッシュ統計:")
        for k, v in cache_stats.items():
            print(f"     {k}: {v}")
        
        latency = report.get('latency', {})
        print(f"\n   レイテンシ:")
        print(f"     平均: {latency.get('avg_ms', 0):.2f}ms")
        print(f"     p95:  {latency.get('p95_ms', 0):.2f}ms")
        print(f"     目標: {latency.get('target_ms', 100)}ms")
        print(f"     ステータス: {latency.get('status', 'N/A')}\n")
        
        # ================== タスク3: 異常検知 ==================
        print("⚠️  【タスク3】異常検知チェック...")
        
        # キャッシュヒット率チェック
        hit_rate = cache_stats.get('hit_rate', 0)
        if hit_rate < 0.5:
            alert = f"⚠️  キャッシュヒット率が低 ({hit_rate:.1%}) - メモリ使用量を確認"
            print(f"   {alert}")
            maintenance_report["alerts"].append(alert)
        
        # レイテンシチェック
        p95_latency = latency.get('p95_ms', 0)
        if p95_latency > 100:
            alert = f"⚠️  p95 レイテンシが目標超過 ({p95_latency:.2f}ms > 100ms)"
            print(f"   {alert}")
            maintenance_report["alerts"].append(alert)
        
        # ドキュメント数チェック
        if rag_doc_count < 100:
            alert = f"⚠️  Vector DB ドキュメント数が少ない ({rag_doc_count} < 100)"
            print(f"   {alert}")
            maintenance_report["alerts"].append(alert)
        
        if not maintenance_report["alerts"]:
            print("   ✅ 異常なし - システム正常稼働中\n")
        else:
            print()
        
        # ================== タスク4: レポート保存 ==================
        print("💾 【タスク4】メンテナンスレポート保存中...")
        report_file = "/tmp/rag_maintenance_latest.json"
        with open(report_file, "w") as f:
            json.dump(maintenance_report, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ レポート保存: {report_file}\n")
        
        # ================== 完了サマリー ==================
        print("=" * 80)
        print(f"✅ 統合メンテナンス完了")
        print(f"   実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   ChromaDB: {chroma_result.get('status', 'unknown')}")
        print(f"   LocalVectorDB: {rag_doc_count} 件")
        print(f"   アラート: {len(maintenance_report['alerts'])} 件")
        print(f"   ステータス: {maintenance_report['status'].upper()}")
        print("=" * 80 + "\n")
        
        return maintenance_report
    
    except Exception as e:
        print(f"❌ メンテナンスエラー: {e}")
        maintenance_report["status"] = "error"
        maintenance_report["error"] = str(e)
        print("=" * 80 + "\n")
        return maintenance_report


# ===== Slack 通知機能

def send_slack_notification(report: dict, webhook_url: str = None):
    """メンテナンスレポートを Slack に通知（オプション）"""
    
    if not webhook_url:
        return  # Webhook URL がない場合はスキップ
    
    import requests
    
    status_emoji = "✅" if report['status'] == 'success' else "⚠️"
    
    message = {
        "text": f"{status_emoji} 統合メンテナンス実行完了",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*統合メンテナンスレポート*\n"
                           f"実行時刻: {report['timestamp']}\n"
                           f"ステータス: {status_emoji} {report['status'].upper()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*ChromaDB*\n{report['chroma_reindex'].get('status', 'unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*LocalVectorDB*\n配置中"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*キャッシュヒット率*\n{report['rag_maintenance'].get('cache_stats', {}).get('hit_rate', 0):.1%}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*p95 レイテンシ*\n{report['rag_maintenance'].get('latency', {}).get('p95_ms', 0):.2f}ms"
                    }
                ]
            }
        ]
    }
    
    try:
        requests.post(webhook_url, json=message)
    except Exception as e:
        logger.error(f"Slack 通知エラー: {e}")


if __name__ == "__main__":
    report = run_daily_maintenance()
    
    # 【オプション】Slack 通知
    # send_slack_notification(report, webhook_url="https://hooks.slack.com/...")
