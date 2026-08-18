# /explore-codebase — リポジトリ構造・依存関係の高速把握

## 使い方
```
/explore-codebase [トピック・キーワード]
```

**例:**
```
/explore-codebase
/explore-codebase Q_risk
/explore-codebase shion agent
```

引数なし: プロジェクト全体の構造サマリーを表示
引数あり: 指定トピックに関連するファイル・シンボルを横断調査

## 処理手順

1. **Serena MCP を優先して使う**（CLAUDE.md の規約通り、Read より先）
   - `get_symbols_overview` でトップレベル構造を把握
   - トピック指定時は `find_symbol` でシンボル名・参照元を検索
   - Serena が使えない場合のみ Glob/Grep にフォールバック

2. **引数なしの場合: 全体構造サマリー**
   - `frontend/src/app/` 配下のページ一覧（25+ページ）
   - `api/main.py` のエンドポイント一覧（`@app.` で始まる行を抽出）
   - `scoring_core.py` の主要関数一覧
   - CLAUDE.md の「要注意領域」表を併記し、危険なファイルを明示

3. **トピック指定時: 関連ファイル横断調査**
   - フロント（`frontend/src/`）・API（`api/`）・スコアリング（`scoring_core.py`, `scoring/`）・
     Vault連携（`obsidian_bridge.py`）の4層それぞれでキーワードを検索
   - 各層でヒットしたファイルパス・行番号・簡単な役割を一覧化

4. **結果を表形式でまとめる**

| 層 | ファイル | 該当箇所 | 役割 |
|---|---|---|---|
| Frontend | `frontend/src/app/.../page.tsx:42` | ... | ... |
| API | `api/main.py:120` | ... | ... |
| Scoring | `scoring_core.py:88` | ... | ... |

5. **要注意領域との重複を警告**
   - ヒットしたファイルが CLAUDE.md の「要注意領域」表に含まれる場合、該当行の注意点を引用して警告表示

## 注意事項
- このコマンドは調査のみ（コード変更は行わない）
- 大規模調査（3クエリ超）が必要な場合は `Explore` エージェントの起動を提案する
- 出力は次の `/plan-feature` や `/write-spec` の入力として使える形式にする
