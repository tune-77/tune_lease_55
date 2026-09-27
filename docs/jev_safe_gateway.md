# Jev Safe Gateway

Jevへコード・diff・社内データをそのまま送らず、限定判断に必要な最小情報だけを渡すためのローカルGateway。

## モード

| mode | 用途 | Jevへ渡るもの |
|---|---|---|
| `abstract` | 私有コード、diff、設計、ログ | 成果物種別、一般化した効果、依頼との関係、リスク区分、検証状態 |
| `aggregate` | 案件・業務データ | 業種・地域・規模のbucket、risk flag、boolean signal |
| `public_excerpt` | 公開コード | 検証可能なGitHub公開URLと最大4,000文字の抜粋 |

私有コード・diffの原文、ファイル名、パス、シンボル名、案件ID、個人情報、秘密情報は送信対象にしない。Gatewayは外部通信を行わず、許可時に出力される `outbound` だけをJevのstateとして使う。

## 実行例

```bash
python scripts/jev_safe_gateway.py --input request.json
```

`request.json` の例:

```json
{
  "mode": "abstract",
  "purpose": "変更候補を依頼との関連度で分類する",
  "items": [
    {
      "source_id": "local-reference-only",
      "artifact_kind": "application code",
      "change_kind": "feature",
      "relation": "direct",
      "effect": "既存画面へ利用者向け操作を追加する",
      "risk_category": "low",
      "verification_status": "focused tests passed"
    }
  ]
}
```

`status=allowed` の時だけ `outbound` をJevへ渡し、返却された `item_1` などの合成IDをローカルの `local_mapping` で元候補へ戻す。`status=blocked` または入力不正なら外部送信しない。

監査ログは既定で `data/jev_safe_gateway_audit.jsonl` に追記される。本文、ローカルID、元データは記録せず、判定、理由コード、件数、outboundのSHA-256だけを残す。

## 安全境界

- `abstract` と `aggregate` は未知フィールドを拒否する。許可項目を増やす時はテストと規約も更新する。
- `aggregate` に生の金額・比率・期間を入れず、ローカル処理でbucketへ変換する。
- `public_excerpt` は `visibility=public`、HTTPS、許可ホスト、秘密情報検査、文字数制限をすべて通す。
- Jevの確率が0.7未満、選択肢外、または投影後の情報で判断不能ならCodex/Claudeへ戻す。
- 削除、公開、mergeなど不可逆・外向きの操作はJevだけで決定しない。
