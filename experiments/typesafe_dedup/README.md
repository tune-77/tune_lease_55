# TypeSafe/Jev 重複判定（Phase 1実測・Phase 2結線）

`scripts/extract_obsidian_improvements.py` の `deduplicate_improvements()` は
段1〜3（完全一致・先頭40文字・サブセット・テーマグループ）と段4（文字2-gram
Jaccard ≥ 0.55）で重複を判定する。段4は日本語の言い換えに弱く、
「PRタイトルにREV番号が入らない」と「`cleanup_improvement_reviews.py --apply` が
空振りする」のように同一原因でも Jaccard ≈ 0.1 になる。閾値を下げると別件まで
併合される。`_THEME_GROUPS` の12グループは、この穴を症状名で手当てした表である。

このパイロットは、灰色帯（Jaccard が段4の閾値未満だが無関係とも言い切れない帯）
だけを Jev の Noul に投げ、閾値を実測で決めるためのもの。

## Phase 1 の境界

- `scripts/extract_obsidian_improvements.py` と `run_daily_improvement_pipeline.sh`
  への差分は **ゼロ**。`typesafe_dedup_guard.py` はどの既存モジュールからも
  import されていない。
- 呼び出し元は `experiments/typesafe_dedup/measure.py` だけ。
- フラグ未設定なら外部通信は一切発生しない。

## Phase 2 の結線

2026-09-20 の実測後、`scripts/extract_obsidian_improvements.py` の決定的な
重複排除の後段へ結線した。運用者承認後、日次処理では既定オンにした。
一時停止する場合は `TYPESAFE_DEDUP_ENABLED=0` を指定する。

- 自動併合は `same_issue >= 0.75` のみ
- 0.40〜0.75未満は自動併合しない
- 企業名・案件番号・メール・電話・金額らしき文字列を含む候補はローカル除外
- Jev失敗時は全候補を別件として残す
- 意味重複排除後が15件以下ならGemini統合を呼ばない

## 実測入力（仕様書からの逸脱）

仕様書 §8 は `~/Library/Logs/tunelease/improvement_YYYYMMDD.log` を入力としていたが、
このログにはパイプライン自身の稼働行（`[改善] auto-improvement-pipeline 実行中...`）
しか入っておらず、候補が存在しなかった。実体は
`reports/improvement_report_YYYYMMDD.json` の `needs_review` / `rejected` で、
各エントリが `{id, title, reason}` を持つ（`tag` は無い）。

母集団の単位にも落とし穴があった。日次レポートは未解決の候補を毎日そのまま
再出力するため、**出現**を数えると1件の候補が数百ペアに膨らむ。既定を直近30
ファイルにしていた初回測定では distinct title が6件しかなく、段1（完全一致）が
1,360ペア全部を確定して灰色帯が0になった。これは仮説の反証ではなく測定窓の
アーティファクト。

もう一つ、そもそも候補ですらない行が混ざっていた。抽出側が `- status: rejected`
のような YAML 行をタイトルとして拾っており、互いに `- status: ` を共有するせいで
Jaccard が偶然この灰色帯に落ちる。`load_candidates()` の
`_is_parse_artifact()` が恒久的に除外する（`--keep-artifacts` で解除可能）。

現在の既定は **全レポート × タイトル重複排除 × パース不良の除外**。
114ファイルから distinct title 192件、うち4件がパース不良で、母集団は188件。

この逸脱は `measure.py` の docstring にも記録してある。

## 実行

```bash
# 帯ごとのペア数だけを見る（オフライン、外部通信なし）
python3 experiments/typesafe_dedup/measure.py

# 灰色帯ペアの全文を表示（§10 の目視確認用）
python3 experiments/typesafe_dedup/measure.py --inspect --width 0.25

# パース不良フィルタを外して母集団を確認する（フィルタ自体の検証用）
python3 experiments/typesafe_dedup/measure.py --keep-artifacts

# 送信（要フラグ。下の「送信前の確認」を先に済ませること）
TYPESAFE_DEDUP_ENABLED=1 TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key \
  python3 experiments/typesafe_dedup/measure.py --send --width 0.25

# 保存済み結果の閾値スイープ（再推論しない）
python3 experiments/typesafe_dedup/measure.py --sweep experiments/typesafe_dedup/results/judged_w25.json
```

環境変数（`typesafe_dedup_guard.py`）:

- `TYPESAFE_DEDUP_ENABLED` — 未設定なら無効。RAG 側の
  `TYPESAFE_RAG_ENABLED` とは独立
- `TYPESAFE_API_KEY` または macOS の `TYPESAFE_API_KEYCHAIN_SERVICE`
- `TYPESAFE_MODEL`（既定 `jev-latest`）
- `TYPESAFE_ENDPOINT`（既定 `https://api.typesafe.ai/v1/systemone`）
- `TYPESAFE_DEDUP_TIMEOUT_SECONDS`（既定 `8`）
- `TYPESAFE_DEDUP_MAX_PAIRS`（既定 `40`、上限 `200`）

API キーをソースコードに書かないこと。送信されるのは改善案の `title` と
`reason` だけで、`tag`・Vault パス・ソースファイル名は意図的に除外している。

## 送信前の確認（仕様書 §10）

改善候補は Obsidian の AI チャットログ由来であり、実在の企業名や案件番号が
混入しうる。`--send` の前に `--inspect` で灰色帯ペアの全文を目視し、混入が
あれば除外すること。この確認を経ずに外部送信しない。

`results/` は既定で `.gitignore` により全無視にしてある。実測結果をコミット
するのは、この目視確認を済ませた後だけ。

### 目視確認の記録

| 日付 | 帯幅 | 確認ペア数 | 除外 | 確認者 |
|---|---|---|---|---|
| 2026-09-20 | 0.25（帯 0.30-0.55） | 33 | 5（パース不良・下記） | tunetune |

実在の企業名・案件番号・金額・個人名は 0 件。固有名詞は `PR #204`（本リポジトリの
GitHub PR 番号）、`蘭丸`（本リポジトリ内の AI エージェント呼称）、`EDINET`、
`Kubernetes` のみで、いずれも外部送信して問題ない。

除外した5ペアは機密ではなく**候補ですらない**。`- status: rejected` のような
YAML 行がタイトルとして取り込まれており、互いに `- status: ` を共有するせいで
Jaccard が偶然この帯に落ちていた。この確認を受けて `load_candidates()` に
`_is_parse_artifact()` を入れ、母集団の段階で恒久的に落としてある
（`--keep-artifacts` で解除して確認できる）。下の指標はフィルタ適用後の数値で、
この表の 33 / 5 は**確認当時に何を見たか**の記録なので更新しない。

## 判定と routing

灰色帯の各ペアに Noul を1つだけ聞く。

- `same_issue` — 同一の根本課題に対する改善案か

独立した判定なので、全ペアを **1リクエスト** にまとめて並列実行する。
閾値の適用は `route_pair()` に分離してあり、重みや閾値を変えても再推論は不要。

運用ポリシーは 3 分岐（仕様書 §8、`measure.py` の `classify_judged_pair()`）:

| `same_issue` | routing |
|---|---|
| ≥ 0.75 | `duplicate` |
| 0.40 〜 0.75 未満 | `needs_human_review` |
| < 0.40 | `distinct` |

誤併合は台帳に痕跡が残らず気づけない。誤分割は重複 REV として見えるので
人が閉じられる。安い方の誤りは見える方なので、自動併合の基準を高く置き、
判断が割れる帯は二択に押し込まず人へ回す。`--sweep` はこの 3 分岐の内訳も
表示する。

TypeSafe が無効・タイムアウト・エラー・不正な確率を返した場合、
`judge_pairs_if_enabled()` は灰色帯を全て「別件」として返す。これは現行の
`deduplicate_improvements()` が灰色帯に対して出す結論と同一であり、
挙動は変わらない。

## 記録する指標（仕様書 §9）

母集団: 全114レポート・タイトル重複排除・パース不良除外後の188候補（2026-09-20 実測）。

| 指標 | 結果 |
|---|---|
| 1. 段1〜3で確定したペア数 / 全ペア数 | 26 / 17,578（theme_group 21・subset 5） |
| 2. 帯幅ごとの灰色帯ペア数（= Jev 呼出し数） | 0.35-0.55: 19 ／ 0.30-0.55: 28 ／ 0.25-0.55: 45 |
| 3. 運用ポリシーの内訳 | duplicate 6 / needs_human_review 19 / distinct 3 |
| 4. 現行 0.55 単独ロジックが取りこぼしていた高信頼重複 | 6ペア（目視で6件とも妥当。正式ラベル評価は未実施） |
| 5. usage トークン | Jev 1.13.0、入力7,581 / 出力554、28ペアを1リクエスト |

仮説の判定: 灰色帯（0.30-0.55）は28ペアで、段4が重複と確定する11ペアの約2.5倍。
現行ロジックが「判断していない」帯が確定分より厚いという、仕様書が狙った穴は
実在する。28ペアは既定の `TYPESAFE_DEDUP_MAX_PAIRS=40` に収まり、1リクエストで
送れる。ただしこの28ペアが本当に重複なのかは人手ラベル（指標3）待ちで、
現時点では**コストと対象数の裏付けまで**。

パース不良フィルタの効果（`--keep-artifacts` との差分）:

| | 候補数 | 全ペア | 段1〜3 | 0.35-0.55 | 0.30-0.55 | 0.25-0.55 |
|---|---|---|---|---|---|---|
| 既定（フィルタ有効） | 188 | 17,578 | 26 | 19 | 28 | 45 |
| `--keep-artifacts` | 192 | 18,336 | 26 | 23 | 33 | 51 |

段1〜3の確定数が26のまま動いていないのが要点で、フィルタは既存の重複構造に
触れず、灰色帯のゴミだけを削っている。除いた4候補が灰色帯に5ペア（帯0.25なら
6ペア）も作っていたのは、無意味な文字列どうしほど Jaccard が中途半端な値に
なりやすいため。送信対象の 15% がこれだった。

注意: 出現ベース（`--keep-repeats`）で数えると、候補2,256件・全254万ペアとなり、
灰色帯（0.30-0.55）は同じ信号が1,572ペアに膨らむ。28ペアの56倍で、コストも
網羅率も2桁過大評価になる。母集団は常にエンティティ単位で取る。

測れないこと: 全体の重複排除率、日次パイプラインへの影響。Phase 1 は
パイプラインに接続していないため。

## 検証

```bash
pytest -q tests/test_typesafe_dedup_guard.py
```

テストは全て `request_fn` を注入するか機能を無効のままにしており、
TypeSafe に接続しない。
