# 審査結果の判断根拠表示

`scoring_core.run_quick_scoring` と純APIのFull審査が、実際に適用した補正・物件所見・審議ゲート・承認ラインの根拠を `judgment_reasons` に記録する。審査画面 `/screening` とウィザード `/lease-kun` に同じ表示を追加する。

判断資産の定義は `static_data/scoring_judgment_assets.json`。`SCORING-*` は計算コードのルールを参照する専用IDであり、既存のcanonical判断資産候補を計算に使用したという意味ではない。定義は現在の `scoring_core.py` の条件と対応する。物件判定は既存の `static_data/useful_life_by_industry.json` と `useful_life_lookup.py` のフォールバック定義を利用する。

各根拠はID・版・タイトル・要約・今回の適用理由・効果区分・実効加減点を保持する。計算条件または説明を変更する場合は該当資産の `asset_version` を上げる。Full審査の既存案件保存にも説明と版を含め、過去の結果を現在の定義で再構築しない。加減点は丸めと0〜100点制限を反映した差分で、物件所見・審議ゲートは0点。

モデル内部の特徴量寄与、追加フィードバック機能、既存ルールの変更は今回の対象外。資産ファイルが読み込めない場合は失敗ログを残して審査を継続する。旧結果・旧Streamlit経路など根拠がない結果には、その旨を表示する。

## 開発環境での確認

リモートに `dev` が存在しなかったため、取得した `master` からローカル `dev` を作成した。本番への反映は行わない。

```bash
pytest tests/test_scoring_logic.py tests/test_scoring_core.py tests/test_scoring_demo_food_service.py tests/test_check_frontend_backend_schema.py
cd frontend
npx tsc --noEmit
npx playwright test e2e/scoring-judgment-basis.spec.ts
```

ブラウザテストは外部APIをモックし、根拠・版・減点・追加減点なしの表示、再読み込み、モバイル幅、旧結果の案内を確認する。APIテストはcalculate/full双方の根拠保持と、Full案件保存への根拠の受け渡しを確認する。
