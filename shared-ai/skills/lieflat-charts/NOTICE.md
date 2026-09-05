# 導入元・利用範囲メモ

- 取込元: https://github.com/larashero3-dotcom/lieflat-charts
- 取込コミット: `eace082a317b696c5570c25826a53a7fa113e984`（2026-09-05時点のdefaultブランチ）
- ライセンス: PolyForm Noncommercial License 1.0.0（`LICENSE`参照）。本リポジトリでの利用は非商用利用であることをユーザーに確認済み。商用転用する場合は別途ライセンス確認が必要。
- 取込範囲: `SKILL.md` / `catalog.md` / `report-catalog.md` / `color-presets.js` / `mono-tokens.js` / `templates/` / `agents/` / `examples/` / `scripts/` のみ。上流の `README*.md` と `docs/assets/`（プレビュー画像・GIF、約19MB）はskillの動作に不要なため除外。プレビューは上流リポジトリを参照。
- 利用範囲の合意事項: 紫苑レビュー・日次レポート生成での利用を想定。審査分析画面（Next.js UI）への直接組込みは対象外（`.claude/rules/frontend.md`でグラフをRecharts固定と規定しており、本skillの単体HTML前提の実装と衝突するため）。UIの配色検討時に`color-presets.js`のトークンだけ参考にするのは可。
