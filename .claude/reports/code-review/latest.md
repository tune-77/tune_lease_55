---
agent: code-reviewer
task: PR #160 — PD除去 & 業種別倒産率ベンチマーク追加 & LGBM成約モデル統合
timestamp: 2026-05-23 12:30
status: partial
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー
PD除去は主要パスで概ね完了しているが、アクティブコード内に複数の残存参照（report_generator.py、batch_scoring.py、floating_bot.py）がある。LGBM統合のロジック自体は堅牢だが単体で呼ばれておらず dead code に近い状態。業種ベンチマークの実装は適切。

## 詳細
- **[components/score_calculation.py:78]** — `_build_learning_pd_result()` が定義されているが、diff 後は呼び出し箇所がゼロ（行1067の `scoring_result` は常に None）。dead code として残存。
- **[components/score_calculation.py:1067]** — `scoring_result` は初期化(None)後に代入される箇所がないため、LGBMモデルの「否決」ペナルティロジックが永遠に発動しない。LGBM統合の実効性がない。
- **[components/score_calculation.py:1270-1275]** — `scoring_output_bridge.json` への書き込みでエラーメッセージに "with PD" と書かれており、コメントがPD除去前のまま。軽微だが残置デバッグ文字列。
- **[report_generator.py:332]** — `res.get("pd_percent", 0)` でPD参照が残存。`res` に `pd_percent` キーが存在しなくなったため `default_prob` は常に0.0になる。審査レポートのリスク文言が機能しない。
- **[components/batch_scoring.py:516-678]** — `grade_to_pd` テーブルと `pd_pct` フィールドが残存。一括スコアリングのエクスポートCSVに "PD概算(%)" 列が出力され続ける。
- **[components/floating_bot.py:33-35]** — `_pd_to_risk()` 関数が残存（ただし呼び出し元は1箇所のみ。つん子コメント選択に pd_percent=0.0 固定で渡される可能性がある）。
- **[shinsa_gunshi_db.py:35/67/72]** — DB スキーマに `pd_pct REAL NOT NULL` が残存。`shinsa_gunshi_ui.py` のスライダーから値を受け取るため UI フローは壊れていないが、主審査フローとは切り離された独立運用になっている。
- **[shinsa_gunshi_logic.py:846/856]** — `success_samples` / `fail_samples` のフォーマット文字列に `s['pd_pct']` が残存。DBから取得した過去事例の `pd_pct` キーが存在しない場合は `KeyError` が発生する。
- **[api/gunshi_gemini.py:58]** — `build_user_prompt()` で `f"PD={pd_pct}%\n"` が削除済みだが、`stream_gunshi_gemini()` 内の `compute_prior(score, pd_pct)` は `pd_pct` が常に0.0で計算される。ベイズ計算の prior が score のみに依存するようになった（意図的なら許容）。
- **[api/main.py:515]** — `GunshiStreamRequest.pd_pct: float = 0.0` は後方互換のため残っており問題ない。ただし `/api/gunshi/chat` の `build_gunshi_prompt` 呼び出しで `pd_pct` 引数を省略した結果、デフォルト 0.0 が渡る（意図通り）。
- **[scoring/predict_one.py:62-75]** — `_LGBM_CONTRACT_CACHE is not None` チェックでは空辞書 `{}` を有効キャッシュとして扱う問題がある。ロード失敗時は `None` を返すため通常は問題ないが、外部から `{}` を注入された場合は壊れる。
- **[scoring/predict_one.py:117-118]** — `op_profit` と `ord_profit` の代入で `_safe_get_float()` を2回呼ぶ（同一引数）。パフォーマンス上軽微だが可読性低下。
- **[data/industry_bankruptcy_bench.py:80-83]** — `OVERALL_AVG_RATE = 0.35` は7業種単純平均で計算上正確。`get_relative_risk()` の閾値（±30%）は妥当。

## 課題・リスク

**重大**
1. `scoring_result` が常に `None` のため、LGBM 統合は `predict_one.py` に実装されたものの `score_calculation.py` から一切呼ばれていない。LGBM の「否決」ペナルティも発動しない。`_build_learning_pd_result()` の呼び出しを復活させるか、直接 `predict_one()` を呼んで `scoring_result` に代入する必要がある。
2. `report_generator.py:332` の `pd_percent` 参照が残存しており、審査レポートのデフォルト確率セクションが常に0%表示になる。レポートの品質に直接影響する。

**軽微**
3. `shinsa_gunshi_logic.py:846/856` の `s['pd_pct']` で KeyError リスク。過去事例DBに `pd_pct` カラムが存在する限りは動作するが、DBリセット後や新規インストール環境で問題になる可能性がある。
4. `scoring_output_bridge.json` への絶対パスハードコード（`/Users/kobayashiisaoryou/clawd/...`）が2箇所に存在。本番・CI 環境移植時に問題になる。
5. `api/main.py:516-517` で GEMINI_API_KEY チェックを削除し `gunshi_gemini.py` 側のフォールバックに移動したが、`HTTPException(503)` による明確なエラーレスポンスが失われた。SSEストリームで代替テキストが返るため UX は維持されるが、監視・アラート目的では503が出なくなる点に注意。

## 後続エージェントへの申し送り
- security-checker: `scoring_output_bridge.json` への絶対パスハードコードが `components/score_calculation.py` に2箇所。パス漏洩ではないが環境依存コードとして要確認。`data/industry_bankruptcy_bench.py` は外部データ（TDB）のハードコード参照のため出典・ライセンス確認を推奨。
- test-runner: `scoring/predict_one.py` の `_load_lgbm_contract_bundle()` と `_build_lgbm_contract_row()` は単体テストが未整備。特に特徴量名の一致チェック（`fmap.get(f, nan)` の網羅性）と NaN 混入時の LGBM 推論動作を確認すること。`data/industry_bankruptcy_bench.py` の `get_bankruptcy_bench()` に対し、未知業種・空文字・プレフィックスのみのケースをテストすること。
