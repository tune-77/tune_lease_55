# 「同じ出来事の別記事か」「同じ要望の重複か」に Jev は効くか（2026-10-02）

判断資産の同一性（AUC 0.86）に続き、答えがはっきりした二値質問として2か所で測った。

## データと正解

- **ニュース**: 業界リスクニュースのノート（同日の見出し＋`## 関連報道`）と err.log の `[news-guard-item]`
  1実行ぶんから、同一実行内で文字Dice≥0.25・埋め込み≥0.80・既存ルールで統合済みのいずれかに当たる見出しペア。
  ユーザー例の振興物産2本（別実行に分かれていた）を追加。116組。
- **改善要望**: `data/cloudrun_improvement_log.jsonl` のチャット改善メモと紫苑の自己提案。状態変更記録
  （「改善ログ: approved」等）は除外。各要望の近傍（文字Jaccard・埋め込みそれぞれ上位1件）から103組。
  相談票（`shion_agent_consultation_queue` / `shion_reasoner_consultation_queue`）は各2行で、ペアが作れなかった。
- 正解は Claude がルーブリックで付けた（同一出来事 = 同じ発表・同じ企業の同じ事象・同じ統計の同じ回／
  同一要望 = 片方を実装すればもう片方も満たされる）。PR #1196 のテストで別物とされた「老舗倒産2本」は別物に合わせた。
  迷ったペア（ニュース10・要望12）は集計から外し、ユーザー確認待ち。
- 判定は REV-424 判定ログへ `mode=offline_label_eval`（guard=`news_same_event` / `improvement_request_dedup`）、
  ラベルは `label_source=assistant:claude_rubric_20261002` で記録した。

## 結果（AUC は 95% ブートストラップCI）

| | Jev | 既存ルール | 文字一致 | 埋め込み |
|---|---|---|---|---|
| ニュース AUC（n=106, 正60） | **0.999** (0.997–1.0) | 0.640 (0.565–0.714) | 0.688 (0.578–0.790) | 0.742 (0.639–0.839) |
| ニュース Brier | 0.039 | 0.396（生）/ 0.228 | 0.239 | 0.216 |
| 改善要望 AUC（n=90, 正34） | **0.992** (0.979–1.0) | 0.570 (0.504–0.648) | 0.858 (0.775–0.931) | 0.848 (0.767–0.920) |
| 改善要望 Brier | 0.086 | 0.333（生）/ 0.232 | 0.199 | 0.181 |

Brier は Jev が生確率、他は1特徴の LOO Platt 補正後（既存ルールは生の0/1も併記）。

- ニュース: 0.6 以上で、既存ルールが取りこぼした同一出来事38組を全て拾い誤統合0。
  既存ルールは別物を4組統合していた（FSA Weekly Review の別号、CGコードの最終化と有識者会議など）。
- 改善要望: 確率が低めに出る（正例の多くが0.35〜0.6）。0.45 以上で誤統合0・正例26/34。
- 注意: 正解ラベルも LLM（Claude）が付けたので、Jev と同じ癖を共有している可能性がある。
  保留ペアへのユーザー回答で補正する。

## 運用への反映（shadow = 記録のみ）

- ニュース: `collect_lease_news_to_obsidian.shadow_same_event_jev` が、既存ルールでまとめた後の代表記事同士
  （Dice≥0.2、最大45組）を Jev に聞き、0.6 以上を `[news-same-event-item]` と判定ログ（mode=shadow）に残す。
  統合は変えない。ニュースガード（`TYPESAFE_NEWS_MODE`）が off の時と `TYPESAFE_NEWS_SAME_EVENT=off` の時は呼ばない。
- 改善要望: `scripts/improvement_request_dedup_shadow.py` を日次パイプライン（post）に追加。新着要望ごとに
  近傍を取り、既存ルールで重複になるものはそのまま、残りだけ Jev に聞いて 0.45 以上を
  `data/improvement_request_dedup_latest.json` に重複候補として出す。改善ログ本体は書き換えない。
- どちらも Jev 不通時は既存ルールのみ（例外は握りつぶして記録だけ）。

## 再現

```bash
TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/same_event_dedup_jev/measure.py news
TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/same_event_dedup_jev/measure.py requests
.venv/bin/python experiments/same_event_dedup_jev/measure.py news --cached   # ラベル修正後の再集計（再推論なし）
```

ラベル付きペアは `data/same_event_dedup_labels_20261002.json`、結果は
`data/same_event_dedup_jev_eval_20261002_{news,requests}.json`（いずれもコミットしない）。
