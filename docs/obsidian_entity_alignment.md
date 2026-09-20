# Obsidian Entity Alignment Standard Workflow

Obsidianノートを「ローカル候補抽出 → Jev意味判定 → 確認済み変更の反映」で整理する標準ワークフロー。分析コマンド自体はVaultを変更しない。

## 標準フロー

1. ローカルで対象ノートを走査し、タイトル、aliases、tags、既存リンクから候補を絞り込む。
2. 外部送信予定の状態を確認し、候補だけをJev/TypeSafeで「別物／関連／重複」に分類する。
3. 低confidenceまたは矛盾候補は人間レビューへ回す。
4. 確認済みの関連候補だけに双方向リンクを追加する。
5. 確認済みの重複候補は正本へ統合し、aliasesと参照元を付け替える。旧ノートは復元可能なArchiveへ移動する。
6. 再スキャンし、候補数、既存リンク検出、テスト結果を確認する。

## 安全境界

- 分析コマンドはVaultを読み取り専用で扱い、リンク追加、ノート統合、移動、削除を行わない。
- 既定対象は `03-知識_業界/`、`リース実務知識/`、`Projects/tune_lease_55/Asset Knowledge/`。
- `Daily`、作業ログ、案件DB、`screening_records`、`past_cases` は既定除外。
- iCloud上で未ダウンロードの `dataless` ノートは、ダウンロードを誘発せずスキップする。
- 候補抽出はローカルのタイトル・aliases・tagsを主に使い、既存リンクも状態確認用の限定候補として扱う。
- 既存リンクだけを根拠に再確認する候補は、通常の意味類似候補を圧迫しないよう最大5組に制限する。
- Jevへ送る場合も絶対パス・Vault内相対パス・既存リンク先・本文抜粋を含めない。
- 外部送信するのはタイトル、aliases、tags、H1〜H3見出しだけ。本文中の案件数値や固有情報は送らない。
- Jev障害時はVaultを変更せず、候補一覧だけを残す。

## 実行

候補抽出だけを行う。APIキーも外部通信も不要。

```bash
python3 scripts/analyze_obsidian_entity_alignment.py
```

外部送信予定の短い状態を目視する。

```bash
python3 scripts/analyze_obsidian_entity_alignment.py --inspect
```

目視確認後にJev判定を実行する。

```bash
TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED=1 \
TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key \
python3 scripts/analyze_obsidian_entity_alignment.py --send
```

出力:

- `reports/obsidian_entity_alignment_latest.md`
- `reports/obsidian_entity_alignment_latest.json`

## 判定

Scoreの3レベル:

1. 別の概念。リンク不要
2. 関連・補完関係。`Related`リンク候補
3. 実質的に同じ知識。重複・統合候補

同じ対象、同じ結論、矛盾のNoulを同じリクエストで取得する。矛盾確率が高いペアと低confidenceのペアは、Scoreに関係なくレビューへ回す。

`duplicate_review`を含む全routeは提案に留め、自動統合しない。Vault変更は判定後の独立した確認済み工程として実施する。
