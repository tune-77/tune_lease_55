# TypeSafe/Jev judgment-asset quality pilot

## What is being measured

`_judgment_asset_quality()` in
`scripts/build_autoresearch_judgment_asset_candidates.py` classifies each
extracted claim as `actionable` or `textbook_general` using keyword-marker
presence only. That verdict feeds `_promotion_status()`, and a
`textbook_general` verdict yields `not_promoted_textbook_general` — the
candidate never reaches a human reviewer and leaves no review trace.

Because suppression is silent, the filter's error rate has never been
observable. This pilot measures it before anything is enabled in production.

## Baseline (corpus of 2026-09-13, 132 candidates)

    suppressed_share            0.765   (101 / 132)
    top reason                  no_case_action_or_condition (82)
    confirmation_question       35 / 37 suppressed (94.6%)

The `confirmation_question` figure identifies a concrete category error rather
than a tuning problem. The rule requires the claim to contain one of
`確認 / 聞 / 質問 / 見る / 照合`:

```python
if candidate_type == "confirmation_question" and not any(
    marker in text for marker in ("確認", "聞", "質問", "見る", "照合")
):
    reasons.append("not_a_confirmation_action")
```

But the extractor emits the confirmation question *itself*, phrased as a
question — `貴社の設備稼働率は業界平均と比較してどの程度ですか？`. A well-formed
question rarely contains the word "確認". The rule asks for a claim that
*describes* an act of confirming, so it rejects almost every genuine
confirmation question it is given.

## Phases

* **Phase 0 (here).** Offline. Establish the baseline, read the suppressed
  claims, confirm the filter misfires. No API call.
* **Phase 1.** `--send` on the existing corpus, then `--compare` the rule and
  Jev verdicts. Decide `AUTO_ACCEPT_MIN` / `MINIMUM_MARGIN` from the observed
  distribution rather than from the provisional defaults.
* **Phase 2.** Enable `TYPESAFE_JUDGMENT_QUALITY_ENABLED` in the daily run.

The guard is not connected until Phase 2: with the flag unset,
`judge_claims_if_enabled()` returns no judgments and every candidate keeps its
rule verdict.

## Usage

```bash
python3 experiments/typesafe_judgment_quality/measure.py               # baseline, offline
python3 experiments/typesafe_judgment_quality/measure.py --inspect 20  # read suppressed claims
python3 experiments/typesafe_judgment_quality/measure.py --send        # Phase 1, calls TypeSafe
python3 experiments/typesafe_judgment_quality/measure.py --compare 'results/judged_*.json'
```

`--send` refuses to run unless `TYPESAFE_JUDGMENT_QUALITY_ENABLED` is set, an
API key resolves, and every class description in
`JUDGMENT_QUALITY_CLASSES` is filled in.

## Safety notes

* The script only reads `data/autoresearch_judgment_asset_candidates.jsonl` and
  writes under `results/`, which is gitignored. It never writes to `data/`.
* Claims are privacy-screened by `is_safe_public_claim()` before any send.
  Money figures are deliberately allowed: published thresholds such as
  補助金上限 are the substance of these claims, not PII.
* A `review` judgment never suppresses a candidate. Suppression is the
  expensive error here; an extra review row is the cheap one.

## リクエスト形状（2026-09-21）

クラス定義は `state.judgment_quality_classes` に **1 回だけ**載せ、各 Choice 質問には
1 行の gloss とポインタだけを持たせる。当初は質問ごとに全文カタログ（約 1.3KB）を
複製していたため、ペイロードがバッチ件数に比例して膨らんでいた。

| 件数 | shared | inline（旧形状） | 削減 |
|---|---|---|---|
| 2 | 2,672 | 2,891 | 8% |
| 16 | 14,078 | 22,725 | 38% |
| 64 | 53,294 | 90,837 | 41% |

`build_quality_request(..., inline_criteria=True)` で旧形状に戻せる。Phase 1 では
**両形状を同一コーパスに投げて判定が一致するか**を先に確認すること。gloss だけで
判定精度が落ちる可能性はオフラインでは検証できない。
