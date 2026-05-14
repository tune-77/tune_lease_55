## 概要
<!-- 何を/なぜ -->

## 対応 SPEC
- spec_id: PX-YYY
- SPEC: `specs/phaseN/PX-YYY-<slug>.md`

## 変更点
- 

## SPEC 準拠チェックリスト
- [ ] PRタイトル / コミットに `spec_id` (PX-YYY) を記載した
- [ ] SPECのAC-xxxに対応するテストを `tests/spec_phaseN/test_PX-YYY.py` に追加しローカルでpassした
- [ ] SPEC外の実装（仕様追加・リファクタ・バグ修正）は含めていない
- [ ] Claude Opus によるSPEC承認（`status: approved`）を確認した

## コミットメッセージ規約
- 形式: `[PX-YYY] <概要>`
- 例: `[P0-001] add PR template`
- 例: `[P1-003] fix snapshot path bug`

## テスト結果
```
$ pytest tests/spec_phaseN/test_PX-YYY.py -v
```

## スクリーンショット / ログ
<!-- 任意 -->

## レビュア向けメモ
- 承認ゲート手順は [`docs/approval_gate.md`](docs/approval_gate.md) を参照
