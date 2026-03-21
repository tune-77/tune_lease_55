# /generate-report — 審査レポート生成

## 使い方
```
/generate-report [--agent] [--pdf]
```
- 引数なし: 現在の session_state から Streamlit UI コンポーネントの改善点を提案
- `--agent`: `report-stylist` エージェントを起動してフルレポートを生成
- `--pdf`: `screening_report.py` の PDF 出力関数を呼び出す

## 処理手順

### デフォルトモード（引数なし）

1. **既存レポートの読み込み**
   以下のファイルを順に Read する：
   - `.claude/reports/agent-team/asset_value_discussion.md`（あれば）
   - `.claude/reports/scoring-audit/latest.md`（あれば）
   - `.claude/reports/report-stylist/latest.md`（あれば）

2. **現在のUI状態をチェック**
   `components/analysis_results.py` の物件スコア表示セクションを Read し、
   以下のコンポーネントが実装されているか確認：
   - [ ] グレードバッジ表示（B-1）
   - [ ] レーダーチャート（B-2）
   - [ ] ウェイト差分テーブル（B-3）
   - [ ] 満了時推定スコア expander（#7）
   - [ ] 補助金カード（subsidy_master）
   - [ ] 推奨リース条件 3カラム

3. **不足コンポーネントを報告してユーザーに選択肢を提示**

```
📊 レポートコンポーネント状況
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ グレードバッジ      — 実装済み
✅ レーダーチャート    — 実装済み
✅ ウェイト差分テーブル — 実装済み
✅ 満了時推定スコア    — 実装済み
✅ 補助金カード       — 実装済み
⚠️ エージェント議論サマリー — 未実装

選択肢:
1. --agent: report-stylist エージェントでフル最適化
2. --pdf: 現状の PDF 出力を確認
3. そのまま終了
```

### `--agent` モード

`report-stylist` エージェントを起動：
```
report-stylist エージェントを起動します。
以下の情報を読んでから、analysis_results.py の
物件スコア表示セクションを最適化してください：
- .claude/reports/agent-team/latest.md
- .claude/reports/scoring-audit/latest.md
```

### `--pdf` モード

1. `screening_report.py` の `build_screening_report_pdf()` 関数を確認
2. テスト用サンプルデータを生成してPDF出力を試みる：
   ```bash
   python3 -c "
   from screening_report import build_screening_report_pdf
   sample = {
     'score': 72.5, 'grade': 'B', 'industry': '製造業',
     'asset_name': 'NC旋盤', 'lease_months': 36,
     'asset_score': 68.0, 'asset_grade': 'B',
   }
   pdf = build_screening_report_pdf(sample)
   with open('/tmp/test_report.pdf', 'wb') as f:
       f.write(pdf)
   print('✅ PDF生成成功: /tmp/test_report.pdf')
   "
   ```
3. 成功した場合: ファイルパスと生成内容の概要を表示
4. 失敗した場合: エラー内容と修正案を提示

## 注意事項
- このコマンドは UI コードを直接変更しない（提案のみ）
- 変更が必要な場合は明示的に確認を取ってから Edit ツールを使う
- `--pdf` モードは `/tmp/` に一時ファイルを生成する（コミット対象外）
