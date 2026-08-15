# Experience Replay Eval Set

- Generated at: `2026-08-14T04:11:56`
- Cases: 10
- Source: `reports/experience_flywheel_latest.json`
- Format: compatible with `scripts/evaluate_answer_quality.py`

## experience_replay_prompt_feedback_01
- Query: 飲食業をやりたいんだけどリース使う？
- Required concepts: 厨房設備/店舗設備, 資金繰り/返済原資, 撤退/中古市場/保全
- Forbidden claims: 飲食業なら一律否決, 開業なら必ず承認
- Require uncertainty: True
- Source: prompt_feedback `xfly_9dbba37a6330947d`

## experience_replay_prompt_feedback_02
- Query: 事業計画はどうやって作る？
- Required concepts: 売上計画/収支計画, 資金繰り/返済原資, 前提/根拠
- Forbidden claims: 計画だけで承認, 根拠不要
- Require uncertainty: False
- Source: prompt_feedback `xfly_ae32ca07e78d4f31`

## experience_replay_prompt_feedback_03
- Query: メイン先の方がリースやりやすいの？
- Required concepts: メインバンク/取引状況, 支援姿勢/融資, 返済原資/資金繰り
- Forbidden claims: メイン先なら無条件で通る, 審査不要
- Require uncertainty: False
- Source: prompt_feedback `xfly_c287a311e04e4b01`

## experience_replay_prompt_feedback_04
- Query: 電車はリース？
- Required concepts: 車両/設備, 長期契約/保守, 資金調達/返済原資
- Forbidden claims: 必ずオフバランス, 会計影響はない
- Require uncertainty: True
- Source: prompt_feedback `xfly_dd58df2f587972ed`

## experience_replay_prompt_feedback_05
- Query: 漁業のリースもあるの？
- Required concepts: 船舶/漁具/設備, 漁獲/季節性/収入変動, 保険/担保/保全
- Forbidden claims: 漁業はリース不可, 季節性は関係ない
- Require uncertainty: True
- Source: prompt_feedback `xfly_cdbc6d411d359296`

## experience_replay_prompt_feedback_06
- Query: 焼却炉はリースできる？
- Required concepts: 許認可/環境規制, 設置場所/撤去費, 保守/処分/残価
- Forbidden claims: 許認可確認は不要, 必ず高く売れる
- Require uncertainty: True
- Source: prompt_feedback `xfly_a605cd449f0831d5`

## experience_replay_prompt_feedback_07
- Query: ダクト付きの空調機はリースできないと聞いた
- Required concepts: 工事費/付帯工事, 物件本体/設備本体, 所有権/原状回復/撤去
- Forbidden claims: 工事費はすべてリース対象, 確認不要, 無条件で対象
- Require uncertainty: True
- Source: prompt_feedback `xfly_4fc62a21883a0d08`

## experience_replay_prompt_feedback_08
- Query: うちのリース会社では工事費は3割くらいまで
- Required concepts: 工事費/付帯工事, 物件本体/設備本体, 所有権/原状回復/撤去
- Forbidden claims: 工事費はすべてリース対象, 確認不要, 無条件で対象
- Require uncertainty: True
- Source: prompt_feedback `xfly_e028b25d80e9c4fb`

## experience_replay_prompt_feedback_09
- Query: 車検切れのくるまはやばいよね
- Required concepts: 車検/登録/法令, 所有者/使用者, 保険/整備/事故
- Forbidden claims: 車検切れでも問題ない, 登録確認は不要
- Require uncertainty: True
- Source: prompt_feedback `xfly_71be22bb41c14ec4`

## experience_replay_prompt_feedback_10
- Query: レンタカーを借りるには
- Required concepts: 車検/登録/法令, 所有者/使用者, 保険/整備/事故
- Forbidden claims: 車検切れでも問題ない, 登録確認は不要
- Require uncertainty: True
- Source: prompt_feedback `xfly_ea1bb05860abab4f`
