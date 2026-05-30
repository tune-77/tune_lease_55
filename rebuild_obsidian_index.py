#!/usr/bin/env python3
"""
Obsidian Vault インデックスをリビルド

Frontmatter 追加後、新しいメタデータをシステムが認識するようにする。
"""

import sys
from pathlib import Path

# モジュール import パス
sys.path.insert(0, str(Path(__file__).parent))

from mobile_app.obsidian_bridge_enhancements import (
    ObsidianIndexWithMetadata,
    prune_stale_cache,
)
from mobile_app.obsidian_bridge import find_vault, _build_vault_index


def rebuild_index():
    """Vault インデックスをリビルド。"""
    vault = find_vault()
    if not vault:
        print("❌ Obsidian Vault が見つかりません")
        return False

    print(f"🔨 Vault インデックスをリビルド: {vault}")
    print()

    # ① 古いキャッシュをクリア
    print("1️⃣  古いファイルハッシュキャッシュをクリア...")
    prune_stale_cache(vault)
    print("   ✓ 完了\n")

    # ② Vault インデックス再構築
    print("2️⃣  Vault インデックスを再構築...")
    _build_vault_index()
    print("   ✓ 完了\n")

    # ③ メタデータ抽出インデックス構築
    print("3️⃣  メタデータ抽出インデックスを構築...")
    try:
        knowledge_paths = [
            p for p in vault.rglob("*.md")
            if not any(d in p.parts for d in ("AI Chat", "Daily", "Improvement Log", "Weekly Review"))
        ]
        print(f"   → {len(knowledge_paths)} 個の知識ノートを検出")

        index = ObsidianIndexWithMetadata()
        index.build(vault, knowledge_paths)
        print("   ✓ BM25 インデックス構築完了\n")

    except Exception as e:
        print(f"   ⚠️  メタデータインデックス構築でエラー: {e}\n")

    print("=" * 60)
    print("✅ インデックスリビルド完了")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. Obsidian アプリを再読み込み（Cmd+R）")
    print("  2. API サーバーを再起動（FORCE_RESTART=1 bash run_next_stable.sh）")
    print("  3. RAG 検索をテスト")

    return True


if __name__ == "__main__":
    rebuild_index()
