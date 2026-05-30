# Phase 1：リアルタイム RAG 統合 実装計画

**実装開始日**: 2026-05-30  
**目標完了日**: 2026-06-13（2週間）  
**対象ユーザー**: 1人（月300チャット）  

---

## 📋 現状分析

### ✅ 既に実装されている機能

```python
# mobile_app/chat_assistant.py:build_chat_reply()
obsidian_hits = collect_obsidian_context(message)      # Line 418
obsidian_digest = build_obsidian_digest(message, obsidian_hits)  # Line 419
web_hits = collect_web_context(message)                 # Line 420
```

**ステータス**: ✅ 毎メッセージで Obsidian 検索が実行されている

### ⚠️ 改善が必要な点

1. **レイテンシが不明**
   - `collect_obsidian_context()` の処理時間を測定していない
   - 目標: < 500ms（ユーザーが体感できない遅延）

2. **キャッシングがない**
   - 同じクエリでも毎回フル検索
   - インデックスファイル IO が毎回発生

3. **インデックス更新の遅延**
   - ユーザーが Vault を編集しても
   - 次のメッセージまで反映されない
   - 最悪で 24時間 遅延

4. **段階的導入の仕組みがない**
   - 全ユーザーに一度に反映
   - 問題があった時に対応できない

---

## 🎯 実装目標

### 目標 1: レイテンシを < 500ms に保つ

```
現在の構成:
  ユーザーメッセージ
    ↓ (collect_obsidian_context)
    → Vault インデックス読み込み
    → BM25 検索実行
    → Wikilink 処理
    ↓ (build_obsidian_digest)
    → 要約生成
    ↓ (Gemini API)
    → LLM に送信
  ↓
  AI 応答

【問題】インデックス IO が毎回発生 → 低速化

【対策】
- Step 1: 処理時間を計測
  → どこが遅いのか特定
  
- Step 2: キャッシング導入
  → 5分以内の同じクエリ = キャッシュ使用
  
- Step 3: インデックス最適化
  → 必要に応じて、メモリ内インデックス検討
```

### 目標 2: インデックス更新を自動化

```
現在:
  Vault ファイル編集
    ↓
  UI から「再構築」を手動実行
    ↓ (最悪24時間遅延)
  インデックス更新

改善後:
  Vault ファイル編集
    ↓
  watchdog (ファイル監視) 自動検知
    ↓ (即座に < 1秒)
  インデックス自動更新
```

### 目標 3: ローリング展開

```
Week 1:
  10% トラフィックで新しい機構をテスト
  → エラーログを監視
  → レイテンシを計測

Week 2:
  50% トラフィックに拡大
  → 大量のチャットでテスト
  → パフォーマンス確認

Week 3:
  100% 本番導入
  → 完全に切り替え
```

---

## 🔧 実装ステップ

### Step 1: レイテンシ計測（Day 1-2）

**ファイル**: `mobile_app/chat_assistant.py`

**実装**:
```python
import time

def build_chat_reply(...):
    start_time = time.time()
    
    # ===== Obsidian コンテキスト取得
    t0 = time.time()
    obsidian_hits = collect_obsidian_context(message) if use_obsidian else []
    t_obsidian = time.time() - t0
    
    # ===== Obsidian 要約生成
    t0 = time.time()
    obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {}
    t_digest = time.time() - t0
    
    # ===== Web コンテキスト取得
    t0 = time.time()
    web_hits = collect_web_context(message) if use_web and _should_search_web(message) else []
    t_web = time.time() - t0
    
    # ===== Gemini 呼び出し
    t0 = time.time()
    # ... Gemini API 呼び出し ...
    t_gemini = time.time() - t0
    
    total_time = time.time() - start_time
    
    # ログ記録
    logger.info(f"Chat latency: obsidian={t_obsidian:.2f}s, digest={t_digest:.2f}s, web={t_web:.2f}s, gemini={t_gemini:.2f}s, total={total_time:.2f}s")
```

**計測ポイント**:
```
- obsidian_hits 取得時間: 目標 < 200ms
- digest 生成時間: 目標 < 100ms
- web_hits 取得時間: 目標 < 100ms
- Gemini API: 目標 < 2秒（既に遅い）
- 合計: 目標 < 500ms
```

**ログ出力先**: `/Users/kobayashiisaoryou/Library/Logs/tunelease/phase1_latency.log`

---

### Step 2: キャッシング機構追加（Day 3-5）

**ファイル**: `mobile_app/chat_assistant.py` に追加

**実装**:
```python
import hashlib
from datetime import datetime, timedelta

class ObsidianContextCache:
    def __init__(self, ttl_seconds: int = 300):
        """TTL: デフォルト 5分"""
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, query: str) -> dict | None:
        """キャッシュから取得（TTL チェック）"""
        key = hashlib.md5(query.encode()).hexdigest()
        entry = self.cache.get(key)
        if not entry:
            return None
        
        if datetime.now() > entry["expires"]:
            del self.cache[key]
            return None
        
        return entry["data"]
    
    def set(self, query: str, data: dict) -> None:
        """キャッシュに保存"""
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = {
            "data": data,
            "expires": datetime.now() + timedelta(seconds=self.ttl),
        }
    
    def invalidate(self) -> None:
        """全キャッシュを無効化（Vault 更新時）"""
        self.cache.clear()

# グローバル インスタンス
_obsidian_cache = ObsidianContextCache(ttl_seconds=300)

def build_chat_reply(...):
    # ===== キャッシュから検索
    cache_key = message.strip()
    cached = _obsidian_cache.get(cache_key)
    if cached:
        obsidian_hits = cached["hits"]
        obsidian_digest = cached["digest"]
        logger.info("使用したキャッシュ")
    else:
        obsidian_hits = collect_obsidian_context(message) if use_obsidian else []
        obsidian_digest = build_obsidian_digest(message, obsidian_hits) if obsidian_hits else {}
        _obsidian_cache.set(cache_key, {"hits": obsidian_hits, "digest": obsidian_digest})
```

**期待効果**:
```
同じクエリの場合:
  1回目: 200ms（フル検索）
  2回目: 5ms（キャッシュ）
  → 96% 削減
```

---

### Step 3: インデックス自動更新（Day 6-8）

**ファイル**: `vault_watcher.py`（既に存在するはず）

**確認**:
```bash
ls -la /Users/kobayashiisaoryou/clawd/tune_lease_55/vault_watcher.py
```

**状態確認コマンド**:
```bash
launchctl list | grep tunelease
tail -f ~/Library/Logs/tunelease/vault-watcher.log
```

**実装内容**:
- ✅ ファイル監視（watchdog）
- ✅ インデックス自動再構築
- ✅ キャッシュ無効化（Vault 編集時）

---

### Step 4: ローリング展開（Day 9-14）

**Week 1: テスト フェーズ（10% トラフィック）**

```python
import random

def should_use_realtime_rag() -> bool:
    """リアルタイム RAG を使うかどうかを判定"""
    # ユーザー ID に基づいて 10% のみ使用
    user_id = "default_user"  # 本来はユーザーセッションから取得
    return hash(user_id) % 100 < 10

def build_chat_reply(...):
    if should_use_realtime_rag():
        # 新しい機構を使用
        obsidian_hits = collect_obsidian_context(message)
        obsidian_digest = build_obsidian_digest(message, obsidian_hits)
        logger.info("新機構を使用")
    else:
        # 従来の機構を使用
        obsidian_hits = collect_obsidian_context(message)  # 同じだが、キャッシュなし
        obsidian_digest = build_obsidian_digest(message, obsidian_hits)
        logger.info("従来の機構を使用")
```

**監視項目**:
- ❌ エラー率: 0% を維持
- ✅ レイテンシ: 平均 300-500ms
- ✅ ユーザー満足度: 主観的フィードバック

**Week 2: 拡大 フェーズ（50% トラフィック）**

```python
def should_use_realtime_rag() -> bool:
    user_id = "default_user"
    return hash(user_id) % 100 < 50  # 50% に変更
```

**Week 3: 本番フェーズ（100% トラフィック）**

```python
def should_use_realtime_rag() -> bool:
    return True  # 全員が新機構を使用
```

---

## 📊 成功基準

### テクニカル
```
✅ レイテンシ: < 500ms（Obsidian 部分）
✅ キャッシュ命中率: > 30%（同じクエリの繰り返し）
✅ エラー率: < 0.1%
✅ インデックス更新遅延: < 1秒（Vault 編集後）
```

### ユーザー体験
```
✅ 「検索が速くなった」と感じる
✅ 「Vault を編集したらすぐ反映される」と感じる
✅ チャット利用回数が増える（現在: 日10回）
```

---

## 📈 期待される改善度

### Before（現状）
```
ユーザー質問
  ↓ (500-1000ms)
Vault 検索（毎回フル）
  ↓ (1-2秒)
Gemini API 呼び出し
  ↓ (2-5秒)
AI 応答

合計: 3.5-8秒
```

### After（改善後）
```
ユーザー質問
  ↓ (5ms - キャッシュ hit / 200ms - miss)
Vault 検索（キャッシング活用）
  ↓ (1-2秒)
Gemini API 呼び出し
  ↓ (2-5秒)
AI 応答

合計: 3.2-7.2秒（5% ～ 10% 削減）

ただし、体感的には「キャッシュが効く頻度」が高いので
「かなり速くなった」と感じる可能性あり
```

---

## 🚀 実装の流れ

### 今日（Day 1）
- [ ] `build_chat_reply()` にレイテンシ計測コード追加
- [ ] 初回実行でログを確認
- [ ] ボトルネック特定

### Day 2-3
- [ ] キャッシング機構の設計
- [ ] `ObsidianContextCache` クラス実装
- [ ] テスト

### Day 4-6
- [ ] ローリング展開の仕組み追加
- [ ] `should_use_realtime_rag()` 実装
- [ ] 統合テスト

### Day 7-14
- [ ] ローリング展開実行
- [ ] Week 1: 10% テスト
- [ ] Week 2: 50% 拡大
- [ ] Week 3: 100% 本番

---

## 💾 コミット計画

```
Commit 1: "Phase 1: Add latency monitoring to chat_assistant"
  - レイテンシ計測機構

Commit 2: "Phase 1: Implement Obsidian context caching"
  - キャッシング機構

Commit 3: "Phase 1: Add rolling deployment for realtime RAG"
  - ローリング展開

Commit 4: "Phase 1: Monitor and optimize latency"
  - パフォーマンス最適化
```

---

## ❓ リスク管理

### リスク 1: レイテンシ増加

**シナリオ**: キャッシング導入で、キャッシュを確認する時間がオーバーヘッドになる

**対策**:
- メモリ内キャッシュを使用（ディスク IO なし）
- ハッシュ検索（O(1)）を使用

---

### リスク 2: 古い情報の混在

**シナリオ**: キャッシュ中に Vault が編集される → 古い情報を返す

**対策**:
- 5分の TTL 設定（十分短い）
- Vault 編集時にキャッシュ無効化

---

### リスク 3: ユーザーが気づかない

**シナリオ**: 改善しても、ユーザーが「速くなった」と感じない

**対策**:
- UI に「Vault から最新情報を取得中...」メッセージ表示
- ログでユーザーに効果を提示

---

## 📞 サポート体制

**何か問題が起きた時**:
1. レイテンシが 1秒 超過 → キャッシング無効化
2. エラー率 1% 超過 → ローリングを 10% に戻す
3. ユーザーフィードバック悪い → 機能停止

**すべて自動で行う仕組みを予定していません**。
ユーザーは 1人なので、直接連絡を取るか、ログを確認してください。

---

## 📋 チェックリスト

- [ ] レイテンシ計測実装
- [ ] 初回実行でログ確認
- [ ] ボトルネック特定
- [ ] キャッシング実装
- [ ] テスト実行
- [ ] ローリング展開コード実装
- [ ] Week 1 テスト（10%）
- [ ] ログ監視（5日間）
- [ ] Week 2 拡大（50%）
- [ ] ログ監視（5日間）
- [ ] Week 3 本番（100%）
- [ ] 効果計測
- [ ] ドキュメント更新

---

**開始日**: 2026-05-30  
**予定終了日**: 2026-06-13
