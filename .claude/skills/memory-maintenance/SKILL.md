---
name: memory-maintenance
description: 日次メモ(memory/YYYY-MM-DD.md)をMEMORY.mdへ棚卸しするスキル。「メモリ整理して」「MEMORY.mdの棚卸し」「メモリメンテナンス」等のキーワード、またはHeartbeat中の定期メンテナンスで使用。
---

# Memory Maintenance (During Heartbeats)

理由: daily files are raw logs while `MEMORY.md` is curated long-term memory.
適用条件: use during heartbeat maintenance and memory review tasks.
削除条件: remove if daily promotion and pruning become fully automated with reliable review evidence.

`memory/YYYY-MM-DD.md` から `MEMORY.md` への自動昇格条件は AGENTS.md の Promotion Triggers を参照。
このスキルは、その自動昇格が拾い切れなかったものを手動で拾うためのチェックである:

1. Read through recent `memory/YYYY-MM-DD.md` files
2. Identify missed significant events, lessons, or insights worth keeping long-term
3. Patch `MEMORY.md` when auto-promotion missed something
4. Remove outdated info from `MEMORY.md` that's no longer relevant

Think of it like a human reviewing their journal and updating their mental model. Daily files are raw notes; MEMORY.md is curated wisdom.
