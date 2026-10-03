#!/usr/bin/env python3
"""比較実験用に、指定した worktree の FastAPI を別ポートで起動する（本番の API・データには触れない）。

使い方（worktree のルートで）:
    OBSIDIAN_VAULT_PATH=<Vaultのコピー> python <このファイル> --port 8101 --counter <file>
    （Gemini の鍵は本番と同じく api.main が worktree の設定ファイルから読む。このスクリプトは鍵に触れない）

- import するのはカレント（worktree）の api.main。data/ や api/chroma_db/ への書き込みは worktree 内のコピーに閉じる
- 実験に関係ない副作用だけ止める: Obsidian 自動保存の判定（Gemini 1回分）と DB の git push
- Gemini の呼び出し回数を数えて --counter に書く（呼び出し1回ごとに1行）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--counter", type=Path, required=True)
    args = parser.parse_args()

    root = Path.cwd()
    sys.path.insert(0, str(root))
    for name in ("USE_GCS_VAULT", "ENABLE_OBSIDIAN_INDEXING", "ENABLE_FEEDBACK_LOADING"):
        os.environ[name] = "false"
    for name in ("K_SERVICE", "CLOUDRUN_PENDING_GCS_ENABLED", "SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN"):
        os.environ.pop(name, None)

    import api.chat_memory as chat_memory
    import api.main as main_module  # ここで worktree の設定ファイル（本番と同じ）から Gemini の鍵が読まれる

    for name in list(os.environ):
        if name.startswith("SLACK"):
            os.environ.pop(name)  # 実験では通知経路を持たせない

    def counted(fn, label):
        def wrapper(*a, **kw):
            with args.counter.open("a", encoding="utf-8") as f:
                f.write(label + "\n")
            return fn(*a, **kw)

        return wrapper

    for name in ("call_gemini_chat", "call_gemini_with_tools"):
        if hasattr(chat_memory, name):
            setattr(chat_memory, name, counted(getattr(chat_memory, name), name))
    for name in ("_auto_save_chat_to_obsidian", "_git_push_db"):
        if hasattr(main_module, name):
            setattr(main_module, name, lambda *a, **kw: None)

    import uvicorn

    uvicorn.run(main_module.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
