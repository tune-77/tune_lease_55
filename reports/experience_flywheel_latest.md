# Experience Flywheel Report

- Generated at: `2026-08-21T19:14:24+00:00`
- Mode: `read_only_experience_flywheel_gate`
- Guardrail: no model weight update, no memory promotion, no skill edit
- Raw candidates: 658
- Deduped candidates: 454
- Duplicates collapsed: 204

## Gate Summary
- observe_only: 31
- promote_to_review: 13
- quarantine: 149
- replay_eval: 261

## Promotion Review
- `screening_feedback` score=15 action=review_screening_judgment_pattern / 境界スコアを、追加確認と条件設定で承認側へ寄せられるか
- `screening_feedback` score=15 action=review_screening_judgment_pattern / 境界スコアを、追加確認と条件設定で承認側へ寄せられるか
- `judgment_asset_feedback` score=14 action=review_judgment_asset_reinforcement / 766ad4e39e346f4f
- `judgment_asset_feedback` score=14 action=review_judgment_asset_reinforcement / 0d0f11e77fba045d
- `screening_feedback` score=13 action=review_screening_judgment_pattern / 承認域だが、競合条件に引っ張られず採算と稟議説明を守れるか
- `screening_feedback` score=12 action=review_screening_judgment_pattern / 承認域の案件を、条件・採算・稟議説明まで崩さず通せるか
- `screening_feedback` score=12 action=review_screening_judgment_pattern / 承認域の案件を、条件・採算・稟議説明まで崩さず通せるか
- `screening_feedback` score=12 action=review_screening_judgment_pattern / 新規・実績薄めの案件を、保全条件と銀行支援で再設計できるか
- `screening_feedback` score=12 action=review_screening_judgment_pattern / 承認域だが、競合条件に引っ張られず採算と稟議説明を守れるか
- `screening_feedback` score=12 action=review_screening_judgment_pattern / 新規・実績薄めの案件を、保全条件と銀行支援で再設計できるか

## Replay Eval
- `judgment_asset_feedback` score=14 action=add_boundary_case_to_eval / b259411afb954d6d
- `prompt_feedback` score=8 action=add_or_update_eval_case / 横須賀に釣りに行く
- `prompt_feedback` score=8 action=add_or_update_eval_case / 飲食業をやりたいんだけどリース使う？
- `prompt_feedback` score=8 action=add_or_update_eval_case / 事業計画はどうやって作る？
- `prompt_feedback` score=8 action=add_or_update_eval_case / メイン先の方がリースやりやすいの？
- `prompt_feedback` score=8 action=add_or_update_eval_case / 電車はリース？
- `prompt_feedback` score=8 action=add_or_update_eval_case / 釣りに行くんだ
- `prompt_feedback` score=8 action=add_or_update_eval_case / 横須賀と横浜は関係あるの？横がついていから
- `prompt_feedback` score=8 action=add_or_update_eval_case / リース以外の話もできるのね
- `prompt_feedback` score=8 action=add_or_update_eval_case / 近くに美味しいお店ある？

## Quarantine
- `shion_experience` score=7 action=do_not_learn / 紫苑は経験によって少し変わる？リース判断ではどう効く？
- `shion_experience` score=7 action=do_not_learn / 紫苑の記憶ループはHermesのMemory.md方式と同じ？
- `shion_experience` score=7 action=do_not_learn / 今何時？
- `shion_experience` score=7 action=do_not_learn / 【審査結果の相談】 境界デモ精機株式会社 ・物件: CNC複合加工機 ・業種: E 製造業 / 24 金属製品製造業 ・営業部: 東京営業部 ・総合スコア: …
- `shion_experience` score=7 action=do_not_learn / リース審査で、数字は悪くないが違和感がある時どう見る？
- `shion_experience` score=7 action=do_not_learn / この案件、条件付き承認にするなら何を確認すべき？
- `shion_experience` score=7 action=do_not_learn / 【審査分析画面からの紫苑レビュー依頼】 この案件を、審査担当者の横にいる紫苑としてレビューしてください。 出力は短く、次の4項目でお願いします。 1. 紫苑の…
- `shion_experience` score=7 action=do_not_learn / 【審査分析画面からの紫苑レビュー依頼】 この案件を、審査担当者の横にいる紫苑としてレビューしてください。 出力は短く、次の4項目でお願いします。 1. 紫苑の…
- `shion_experience` score=7 action=do_not_learn / 【審査分析画面からの紫苑レビュー依頼】 この案件を、審査担当者の横にいる紫苑としてレビューしてください。 出力は短く、次の4項目でお願いします。 1. 紫苑の…
- `shion_experience` score=7 action=do_not_learn / ものづくり補助金はリースでも使える？注意点は？
