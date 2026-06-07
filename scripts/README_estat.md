# e-Stat 業種別統計 動的更新フレーム

## 概要

`fetch_estat_industry.py` は e-Stat API（日本政府統計ポータル）から  
業種別の営業利益率を取得し、`static_data/industry_benchmarks.json` の `op_margin` を自動更新します。

API キーがない環境ではスキップするため、**開発環境・CI での実行に影響しません**。

---

## e-Stat API キーの取得方法

1. https://www.e-stat.go.jp/api/ にアクセス
2. 「ユーザー登録」→ メールアドレスで無料登録
3. ログイン後「マイページ」→「API 機能（アプリケーション ID）」→「発行」
4. 発行された `appId`（英数字の文字列）をコピー

登録・利用は**完全無料**。商用利用も可能（CC BY 4.0）。

---

## 環境変数の設定方法

### 一時設定（ターミナルセッション内）

```bash
export ESTAT_APP_ID=your_app_id_here
python3 scripts/fetch_estat_industry.py
```

### 永続設定（~/.zshrc）

```bash
echo 'export ESTAT_APP_ID=your_app_id_here' >> ~/.zshrc
source ~/.zshrc
```

### LaunchAgent での設定（自動実行用）

`~/Library/LaunchAgents/com.tunelease.improvement-pipeline.plist` に追記:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>ESTAT_APP_ID</key>
    <string>your_app_id_here</string>
</dict>
```

---

## 取得できるデータ一覧

| データ名 | 統計名 | statsDataId | 更新頻度 |
|----------|--------|-------------|----------|
| 業種別売上高・営業利益 | 法人企業統計調査（財務省） | 0003084108 | 年1回（3月頃） |

### 対応業種マッピング

| e-Stat 大分類コード | industry_benchmarks.json キー |
|--------------------|-------------------------------|
| D（製造業・食料品） | 09 食料品製造業 |
| E06（金属製品） | 21 金属製品製造業 |
| E07（生産用機械） | 24 生産用機械器具製造業 |
| E08（情報通信機械） | 26 情報通信機械器具製造業 |
| F（建設業） | 06 総合工事業 |
| G（卸売業） | 50-55 各種卸売業 |
| H（小売業） | 56-61 各種小売業 |
| I（運輸業） | 44 道路貨物運送業 |
| J（飲食・宿泊） | 76 飲食店 |
| K（不動産業） | 68 不動産代理・仲介 |
| L（物品賃貸業） | 70 物品賃貸業(リース・レンタル) |
| N（サービス業） | 91 職業紹介・労働者派遣業 |
| P（医療・福祉） | 83 医療業(病院・診療所) |
| Q（社会保険・介護） | 85 社会保険・社会福祉・介護事業 |

---

## 出力ファイル

| ファイル | 内容 |
|----------|------|
| `static_data/industry_estat_cache.json` | 最終取得時刻・業種別マージン生データ（差分確認用） |
| `static_data/industry_benchmarks.json` | `op_margin` フィールドが e-Stat 実績値で更新される |

`industry_benchmarks.json` の更新されたエントリには以下フィールドが追加されます:

```json
{
    "op_margin": 4.2,
    "op_margin_source": "e-Stat法人企業統計",
    "op_margin_updated": "2026-06-07"
}
```

---

## 手動実行

```bash
cd /Users/kobayashiisaoryou/clawd/tune_lease_55
export ESTAT_APP_ID=your_app_id_here
python3 scripts/fetch_estat_industry.py
```

キーなしでの動作確認（スキップを確認）:

```bash
unset ESTAT_APP_ID
python3 scripts/fetch_estat_industry.py
# → [fetch_estat_industry] ESTAT_APP_ID 未設定 → スキップ
```

---

## 自動実行

`run_daily_improvement_pipeline.sh` の Step 10 として毎朝 AM 4:00 に実行されます。  
`ESTAT_APP_ID` が設定されていない場合は自動スキップされます。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `status=100` エラー | appId が間違い / 未登録 | e-Stat マイページで appId を再確認 |
| `取得した業種別マージン数: 0` | statsDataId の構造変更 | e-Stat でデータセット構造を手動確認 |
| タイムアウト | ネットワーク問題 | `|| true` でスキップ済み、次回実行を待つ |
