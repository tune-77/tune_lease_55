#!/usr/bin/env python3
"""
Obsidian Vault ファイルシステム監視

.md ファイルの変更を自動検知して、インデックスをリアルタイム更新。
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
except ImportError:
    print("⚠️  watchdog が必要です。インストール: pip install watchdog")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
VAULT_ROOT = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault"
PROJECT_ROOT = Path.home() / "clawd" / "tune_lease_55"
LOGS_DIR = Path.home() / "Library" / "Logs" / "tunelease"


class VaultChangeHandler(FileSystemEventHandler):
    """Vault 内のファイル変更を検知・処理。"""

    def __init__(self):
        self.last_reindex = time.time()
        self.reindex_cooldown = 30  # 30秒のクールダウン（重複実行防止）

    def on_modified(self, event: FileModifiedEvent):
        """ファイル変更時のハンドラ。"""
        # .md ファイルのみ対応
        if not event.src_path.endswith(".md"):
            return

        # チャットログ・AI生成ファイルは除外（高頻度変更）
        excluded_dirs = {
            "AI Chat",
            "Daily",
            "Improvement Log",
            "Weekly Review",
            "Generated",
        }
        if any(excluded in event.src_path for excluded in excluded_dirs):
            return

        # クールダウンチェック（重複リインデックス防止）
        now = time.time()
        if now - self.last_reindex < self.reindex_cooldown:
            return

        self.last_reindex = now

        # ファイルパス
        rel_path = Path(event.src_path).relative_to(VAULT_ROOT)
        logger.info(f"📝 変更検知: {rel_path}")

        # インデックス更新を非同期で実行
        self.trigger_reindex()

    def trigger_reindex(self):
        """インデックスリビルドをトリガー。"""
        try:
            logger.info("  → インデックス更新を開始...")
            result = subprocess.run(
                ["python3", str(PROJECT_ROOT / "rebuild_obsidian_index.py")],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("  ✅ インデックス更新完了")
                self._log_event("index_rebuilt", "success")
            else:
                logger.error(f"  ❌ インデックス更新失敗: {result.stderr}")
                self._log_event("index_rebuilt", "failed")
        except subprocess.TimeoutExpired:
            logger.error("  ❌ インデックス更新がタイムアウト")
            self._log_event("index_rebuilt", "timeout")
        except Exception as e:
            logger.error(f"  ❌ エラー: {e}")
            self._log_event("index_rebuilt", "error")

    @staticmethod
    def _log_event(event_type: str, status: str):
        """イベントをログに記録。"""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ledger = LOGS_DIR / "ledger.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "status": status,
        }
        try:
            ledger.write_text(
                ledger.read_text() + json.dumps(entry, ensure_ascii=False) + "\n"
            )
        except Exception as e:
            logger.error(f"ログ記録エラー: {e}")


def main():
    """メイン処理: Vault 監視を開始。"""
    if not VAULT_ROOT.exists():
        logger.error(f"❌ Vault が見つかりません: {VAULT_ROOT}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Obsidian Vault ファイルシステム監視")
    logger.info("=" * 60)
    logger.info(f"Vault: {VAULT_ROOT}")
    logger.info("監視開始... （Ctrl+C で終了）\n")

    # Observer を起動
    event_handler = VaultChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, str(VAULT_ROOT), recursive=True)

    try:
        observer.start()
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n\n監視停止...")
        observer.stop()
        observer.join()
        logger.info("✅ 終了しました")


if __name__ == "__main__":
    main()
