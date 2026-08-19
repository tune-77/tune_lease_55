# /plan-feature — 実装前の作業計画整理（Plan-First Checkpoint）

CLAUDE.md の **Plan-First Checkpoint** ルールを実行可能にしたコマンド。
`scoring_core.py` / `analysis_*.py` / フロントエンドコンポーネント等、要注意領域を
変更し始める前に必ずこれを通すこと。

## 使い方
```
/plan-feature <実装したい機能・変更の説明>
```

**例:**
```
/plan-feature Q_risk が60超の案件に自動で強警戒バッジを表示したい
```

## 処理手順

1. **CLAUDE.md と関連ルールを読む**
   - `CLAUDE.md` の「要注意領域」表を確認し、対象ファイルが該当するか判定
   - 該当する場合 `.claude/rules/frontend.md` / `workflow.md` / `security.md` を追加で読む
   - 対象が `scoring_core.py` など数値ロジックに関わる場合、`APPROVAL_LINE` 等の既存定数を
     import 済みか確認する前提で計画する（別定数のハードコード禁止）

2. **出典を確認する（Freshman Rules: Cite the Source）**
   - スコアリング・審査ロジック・業種データに関わる変更なら `static_data/` または `notes/` を検索し、
     根拠ファイル名を特定する
   - 出典が見つからない場合は計画に「これは推測です」と明記する

3. **矛盾チェック（Freshman Rules: Kill the Assumptions）**
   - `ledger.jsonl` の直近3ヶ月以内の関連エントリを検索し、矛盾する決定がないか確認
   - 矛盾があれば実装前にユーザーへ確認する

4. **3文の変更計画を提示する**
   - 1文目: 何を変更するか（対象ファイル・関数）
   - 2文目: どう変更するか（アプローチ）
   - 3文目: 完了条件（何が✅になれば完了か）

5. **曖昧さが残る場合は着手しない**
   - CLAUDE.md 記載の通り「○○が✅になれば完了と理解しましたが合っていますか？」と
     ユーザーに確認してから1行も書かない

## 出力例
```
## 変更計画
1. `frontend/src/app/cases/[id]/page.tsx` に Q_risk バッジ表示ロジックを追加する
2. `scoring_core.py` からインポート済みの Q_risk 閾値（≥60=強警戒）を参照し、
   既存の credit_quantum_strong_warning フラグに応じて赤バッジを条件レンダリングする
3. 完了条件: Q_risk≥60 の案件詳細ページで「強警戒」バッジが表示されれば完了

根拠: notes/quantum_risk_thresholds.md（閾値定義）
確認: 上記の完了条件で合っていますか？
```

## 注意事項
- このコマンドはコードを書かない（Read-only な調査・確認のみ）
- 承認を得るまで Edit/Write ツールは使わない
- 対象が要注意領域外の軽微な変更であっても、3文計画の提示自体はスキップしない
