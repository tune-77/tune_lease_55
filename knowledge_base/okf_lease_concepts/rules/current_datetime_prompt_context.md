---
type: agent_policy
title: AI回答の現在日時コンテキスト
domain: agent
tags: [AIチャット, 現在日時, プロンプト, RAG]
source: implementation
confidence: high
status: active
updated: 2026-06-20
related:
  - api/context/time_context.py
  - ai_chat.py
  - api/gunshi_gemini.py
  - mobile_app/chat_assistant.py
---

# AI回答の現在日時コンテキスト

## 要点

AIチャット、軍師AI、モバイルチャットのプロンプトには、JSTの現在日時を明示する。相対日付や「今日」「最近」への回答で古い年月を前提にしないための共通ルール。

## 判断ルール

- プロンプトに `【現在日時】` がなければ追加する。
- 同じプロンプトに二重追加しない。
- 日付は `YYYY年MM月DD日 HH:MM (JST, YYYY-MM-DD, 曜日)` の形で保持する。

## 実装

- `api/context/time_context.py`
- `ai_chat.py`
- `api/gunshi_gemini.py`
- `mobile_app/chat_assistant.py`

