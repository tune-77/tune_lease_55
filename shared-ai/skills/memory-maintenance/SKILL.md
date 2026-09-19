---
name: memory-maintenance
description: 日次メモ(memory/YYYY-MM-DD.md)をMEMORY.mdへ棚卸しするスキル。「メモリ整理して」「MEMORY.mdの棚卸し」「メモリメンテナンス」等のキーワード、またはHeartbeat中の定期メンテナンスで使用。
---

# Memory Maintenance (During Heartbeats)

理由: daily files are raw logs while `MEMORY.md` is curated long-term memory.
適用条件: use during heartbeat maintenance and memory review tasks.
削除条件: remove if daily promotion and pruning become fully automated with reliable review evidence.

## Access Guard

`MEMORY.md` may contain personal context. Read or edit it only in a private main session
(a direct conversation with the user). In Discord, group chats, shared sessions, or sessions
with other people, do not open or modify `MEMORY.md`; stop this skill without exposing its contents.

日次メモから長期記憶へ昇格する候補は、次のいずれかを満たすものに絞る:

- 同種の課題や質問へ3回以上対応した
- 今後の意思決定に影響する方針変更があった
- 再発防止したい失敗や注意点が発生した
- 1週間以上有効と見込める個人設定や好みが確認できた

昇格時は「事実・影響・次の行動」を短くまとめ、個人情報や秘密情報は必要最小限にする。
このスキルは、自動昇格が拾い切れなかったものを手動で拾うためのチェックである:

1. Read through recent `memory/YYYY-MM-DD.md` files
2. Identify missed significant events, lessons, or insights worth keeping long-term
3. Patch `MEMORY.md` when auto-promotion missed something
4. Remove outdated info from `MEMORY.md` that's no longer relevant

Think of it like a human reviewing their journal and updating their mental model. Daily files are raw notes; MEMORY.md is curated wisdom.
