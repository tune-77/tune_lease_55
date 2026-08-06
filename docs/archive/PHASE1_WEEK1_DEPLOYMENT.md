# Phase 1 Week 1: テスト開始 📋

**開始日**: 2026-05-30（本日）  
**テスト期間**: 2026-05-31 ～ 2026-06-07（7日間）  
**トラフィック比率**: 10%  
**ステータス**: ✅ **デプロイメント開始**

---

## 🚀 デプロイメント手順

### Step 1: 本番環境への反映確認

```bash
# Git の最新コミットを確認
git log --oneline -5

# Expected output:
# 35a7b12 Phase 1: Production ready - all safeguards implemented
# 95b68b2 Phase 1: Add critical production safeguards
# 6d27b6b Phase 1 implementation status: Step 1-2 complete
# c969f11 Phase 1 Step 2: Implement Obsidian context caching
# 537bccc Phase 1 Step 1: Add latency monitoring to chat_assistant
```

### Step 2: 本番環境への適用

本番環境は以下のいずれかの方法で最新コードを適用：

**方法 A: Git pull（推奨）**
```bash
cd /Users/kobayashiisaoryou/clawd/tune_lease_55
git pull origin master
```

**方法 B: Docker rebuild（使用している場合）**
```bash
docker build -t tune-lease-55:latest .
docker run -e ENABLE_OBSIDIAN_CACHE=true ...
```

### Step 3: 環境変数確認

```bash
# キャッシングが有効になっているか確認
echo $ENABLE_OBSIDIAN_CACHE
# Expected: "true" または 未設定（デフォルトで有効）
```

### Step 4: サービス再起動

```bash
# AI Chat サービスを再起動
# 方法はホスティング環境に応じて異なります

# 例：systemctl の場合
sudo systemctl restart tune-lease-55-ai-chat

# 例：docker-compose の場合
docker-compose restart ai-chat

# 例：Streamlit の場合
pkill -f streamlit
streamlit run /path/to/app.py &
```

### Step 5: ログ監視体制の開始

```bash
# リアルタイムログ監視（ターミナル 1）
tail -f ~/Library/Logs/tunelease/phase1_latency.log

# キャッシュ統計監視（ターミナル 2）
tail -f ~/Library/Logs/tunelease/phase1_latency.log | grep CACHE_STATS

# エラー監視（ターミナル 3）
tail -f ~/Library/Logs/tunelease/phase1_latency.log | grep -i error
```

---

## 📊 Week 1 テスト計画

### 監視項目（毎日確認）

**1️⃣ レイテンシ指標**
```bash
python3 scripts/analyze_phase1_logs.py
```

✅ **チェック項目**:
- [ ] Obsidian 検索: 150ms 前後（< 500ms）
- [ ] Obsidian 要約: 50ms 前後
- [ ] Total: 1.5-2.0s 程度
- [ ] **改善前比**: 計測開始（基準値なし）

**2️⃣ キャッシュ指標**
```bash
tail -f ~/Library/Logs/tunelease/phase1_latency.log | grep cache_status | head -100 | \
  awk -F'cache_status=' '{print $2}' | sort | uniq -c
```

✅ **チェック項目**:
- [ ] cache_hit: 30% 以上
- [ ] cache_miss: 40% 程度
- [ ] cache_disabled: 10-20% 程度

**3️⃣ エラー指標**
```bash
grep -i error ~/Library/Logs/tunelease/phase1_latency.log | wc -l
```

✅ **チェック項目**:
- [ ] エラー件数: 0 件（目標）
- [ ] cache_error_fallback: いくつか出る場合は要対応

---

## 📅 Week 1 テストスケジュール

### Day 1（2026-05-31）: デプロイ + 初期確認
```
09:00 - デプロイメント実施
09:30 - ログ監視開始
10:00 - 初期動作確認
        ✅ チャット応答確認
        ✅ ログ出力確認
        ✅ エラーなし確認
12:00 - 初期レポート作成
```

### Day 2-7（2026-06-01～06-07）: 毎日監視

**毎朝 8:00**:
```bash
# 日次レポート生成
python3 scripts/analyze_phase1_logs.py > ~/Reports/phase1_day_X.txt

# 重要な指標確認
tail -20 ~/Reports/phase1_day_X.txt
```

**毎夕方 18:00**:
```bash
# エラー件数確認
grep -i error ~/Library/Logs/tunelease/phase1_latency.log | tail -5

# キャッシュ効率確認
grep CACHE_STATS ~/Library/Logs/tunelease/phase1_latency.log | tail -1
```

---

## 🎯 Week 1 終了時の判定基準

### ✅ Week 2 へ進む条件（全て達成）

```
✅ エラー率: 0% を維持
   └─ Requirement: エラー件数 = 0

✅ レイテンシ削減: >= 5%
   └─ 計測開始のため、基準値がない場合は「計測完了」で OK

✅ キャッシュ hit 率: >= 30%
   └─ Requirement: (cache_hit / total_requests) >= 30%

✅ ロールバック確認: 問題なく無効化できる
   └─ export ENABLE_OBSIDIAN_CACHE=false で即座に無効化可能
```

### ❌ Week 2 を延期する条件（いずれか該当）

```
❌ エラー率が 1% 以上
   └─ 原因調査 → 修正 → 再テスト

❌ キャッシュ hit 率が 20% 未満
   └─ 原因分析（クエリ多様性など）

❌ ロールバック失敗
   └─ 即座にロールバック実行
```

---

## 🛡️ 緊急対応フロー

### パターン 1: エラーが多く発生

```
発見 → 即座にロールバック → 原因調査 → 修正 → 再テスト

# ロールバック実行（即座）
export ENABLE_OBSIDIAN_CACHE=false

# サービス再起動
sudo systemctl restart tune-lease-55-ai-chat

# ユーザー確認
# 「チャットは動いているか？」を確認
```

### パターン 2: メモリ使用量が増加

```
# 監視スクリプトで確認
python3 scripts/analyze_phase1_logs.py | grep "size_limit_evictions"

# 正常な動作（LRU で自動削除）→ 何もしない
# 異常（メモリリーク）→ ロールバック
```

### パターン 3: キャッシュヒット率が低い

```
原因: クエリが毎回異なる（正常動作）
対応: そのまま継続（5-10% は許容範囲）
```

---

## 📝 Day 1 チェックリスト

**デプロイメント前**:
- [ ] 全コミット確認（5個）
- [ ] ロールバック手順を確認
- [ ] 監視スクリプトが動作するか確認
- [ ] ログディレクトリが存在するか確認

**デプロイメント**:
- [ ] `git pull` または Docker rebuild
- [ ] 環境変数確認（ENABLE_OBSIDIAN_CACHE=true）
- [ ] サービス再起動
- [ ] ログ監視開始

**初期確認**:
- [ ] AI Chat が応答する
- [ ] ログが出力される
- [ ] エラーがない
- [ ] キャッシュが動作している

**初期レポート作成**:
```bash
python3 scripts/analyze_phase1_logs.py > ~/Reports/phase1_day1_initial.txt
cat ~/Reports/phase1_day1_initial.txt
```

---

## 📊 期待される初期データ

### 最初の 1-2 時間

```
Total requests: 50-100

Cache Status:
├─ cache_hit: ~20% (キャッシュが効き始める)
├─ cache_miss: ~70% (まだ新規クエリが多い)
└─ cache_disabled: ~10% (一部ユーザーはオプトアウト可能)

Latency:
├─ obsidian_search: 150-200ms
├─ obsidian_digest: 40-60ms
├─ web_search: 20-40ms
└─ total: 1.5-2.0s

Status: ✅ 正常（初期段階）
```

---

## 🚨 異常判定ライン

**即座にロールバック要因**:
```
❌ エラー率 > 1%
❌ メモリ使用量 > 1GB
❌ チャットが応答しない（timeout）
❌ キャッシュで無限ループ
```

**様子見（48 時間まで）**:
```
⚠️  キャッシュ hit 率 20%
⚠️  レイテンシが改善していない
⚠️  キャッシュエラーが時々出ている
```

---

## 📞 連絡先・対応方法

**問題発生時**:
```
1. ログを確認
2. 原因を特定
3. ロールバック（必要に応じて）
4. 原因調査・修正
5. 再テスト
```

---

## ✅ Week 1 開始チェック

```
✅ コード実装: 完了
✅ テスト: 成功
✅ ドキュメント: 完成
✅ ロールバック手段: 準備完了
✅ 監視体制: 準備完了
✅ デプロイメント手順: 確認済み
✅ 判定基準: 明確化完了

🚀 Week 1 テスト開始準備完了
```

---

**開始時刻**: 2026-05-30 現在（本日）  
**テスト期間**: 7日間  
**判定日**: 2026-06-08

