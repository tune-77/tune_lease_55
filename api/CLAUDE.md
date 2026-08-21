# api/ — ローカルルール

理由: `api/`単体で100ファイル超（chat_*パイプライン、shion_*エージェント群、routers/、game_theory/、knowledge/、crystallizer/、context/）が混在し、ルートCLAUDE.mdの要注意領域表だけでは経路混同・import事故の再発防止に不十分なため。
適用条件: `api/`配下のファイルを変更する時。
削除条件: ここに書いた注意点が自動テスト・lintで機械的に担保された時。

## main.py `/api/chat` — 経路混同に注意

`intent`分岐で3経路（改善/通常/軍師AI）が混在している（`api/main.py`内 `req.intent` 参照箇所を grep で確認: `chat_intent.py`のガイダンス構築、`intent == "improvement"`分岐等）。分岐を跨いだ共通化・リファクタは経路を壊しやすいので、変更前に該当`intent`値ごとの挙動を確認すること。ルートCLAUDE.mdの表も参照。

## shion_*.py（ADKエージェント群）— import方針がファイルごとに違う

- `shion_agent.py`（本流・`api/routers/gunshi.py`のストリーミング経路から呼ばれる）: `google.adk`を**モジュール先頭でimport**する方針。ツールが継続的に追加されるため。
- `shion_debate_adk.py`（フォールバック専用・凍結・ツール無し）: `google.adk`を**モジュール先頭でimportしない**（遅延import）。`google.adk`未導入環境でも動く必要があり、あらゆる例外を握ってスコア由来の最小結果へ劣化する設計。

両者は設計方針が異なるため統合しない。新規`shion_*.py`を追加する際は、どちらの方針に合わせるか意識すること。

## サブディレクトリの役割

- `routers/`: FastAPIルーター群（エンドポイント定義）。新規エンドポイントは既存の対応するルーターファイルに追加し、`main.py`に直接生やさない。
- `game_theory/`, `context/`, `crystallizer/`, `knowledge/`: それぞれ独立した機能単位。跨ぐ変更は影響範囲が広がりやすいので、変更前に依存関係を確認する。

## 参照

- スコア閾値・要注意領域の全体表はルート`CLAUDE.md`を参照（重複記載しない）
- エージェント間のレポート受け渡しは`.claude/AGENTS.md`を参照
