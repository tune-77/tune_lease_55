# PHASE 3 企画書: ユーザーフィードバック機構の詳細設計

**企画日**: 2026-05-30  
**対象**: PHASE 2 完了後の実装  
**工数見積**: 2-3 週間  
**優先度**: ⭐⭐（精度継続改善の基盤）

---

## 📋 概要

PHASE 1・2 で実現した **Hybrid Search（精度 95%）** を、ユーザーフィードバックで **継続的に改善** する仕組みを構築。

### 目標

| 項目 | 現状 | 目標 | 期限 |
|------|------|------|------|
| **検索精度** | 95% | 98% | 6ヶ月 |
| **フィードバック蓄積** | 0件 | 1,000件/月 | 3ヶ月 |
| **Weights 自動調整** | 固定 (0.6:0.4) | 動的調整 | 4ヶ月 |

---

## 🎯 システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                      ユーザー                            │
│              (チャット UI 画面)                          │
└────────────────┬──────────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │  検索結果       │
         │ ┌────────────┐ │
         │ │ 👍  👎    │ │ ← フィードバック UI
         │ └────────────┘ │
         └───────┬────────┘
                 │ (フィードバック送信)
    ┌────────────▼──────────────┐
    │ FeedbackCollector         │ ← 新規実装
    │ ├─ 非同期記録             │
    │ └─ バッチ処理            │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │  feedback_log.jsonl               │
    │  (ユーザーフィードバックログ)       │
    └────────────┬───────────────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │  FeedbackAnalyzer                │ ← 新規実装
    │  ├─ 日次集計                      │
    │  ├─ クエリ別精度分析              │
    │  └─ Weights 提案                  │
    └────────────┬───────────────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │ Hybrid Search（動的Weights）     │
    │ semantic_weight = ? (動的)        │
    │ bm25_weight = ? (動的)           │
    └───────────────────────────────────┘
```

---

## 🎨 UI/UX 仕様

### 1. フィードバック UI コンポーネント

#### 配置: チャット画面の検索結果下

```
┌─────────────────────────────────────────────┐
│ 検索クエリ: "飲食業リース"                    │
├─────────────────────────────────────────────┤
│ 【結果 1】                                   │
│ タイトル: 飲食業の赤字リース対応             │
│ 抜粋: 売上赤字で 3 年連続損失の場合...       │
│                                              │
│ この結果は役に立ちましたか？                 │
│ ┌────────────┬──────────────┐               │
│ │   👍       │      👎      │               │
│ │  役立った  │  役に立たず  │               │
│ └────────────┴──────────────┘               │
│ フィードバックありがとうございます ✅       │
├─────────────────────────────────────────────┤
│ 【結果 2】                                   │
│ ...
└─────────────────────────────────────────────┘
```

#### UI コンポーネント詳細

**ボタン仕様:**
- **サイズ**: 40px × 40px
- **アイコン**: 👍 / 👎 (emoji or SVG)
- **状態**: 
  - 未選択: グレー (#999)
  - 選択済み: 緑/赤
  - ホバー: ツールトip 表示
- **アニメーション**: クリック時にバウンス（200ms）

**フィードバック送信フロー:**
```
ユーザークリック
    ↓
UI 状態変更（即座に色が変わる）
    ↓
バックグラウンド送信（非同期）
    ↓
確認メッセージ表示（「フィードバックを保存しました」）
    ↓
3 秒後に消える
```

### 2. フィードバック履歴表示（オプション）

チャット画面にサイドパネルで「フィードバック履歴」を表示（実装 Phase 1）

```
フィードバック履歴
├─ 本日: 12 件
│  ├─ 👍 8 件
│  ├─ 👎 4 件
│  └─ 精度: 67%
├─ 昨日: 18 件
│  ├─ 👍 15 件
│  ├─ 👎 3 件
│  └─ 精度: 83%
└─ 先週: 94 件
   ├─ 👍 79 件
   ├─ 👎 15 件
   └─ 精度: 84%
```

### 3. 管理画面（ダッシュボード）

**URL**: `/dashboard/feedback-analysis`

```
┌─────────────────────────────────────────────┐
│          フィードバック分析ダッシュボード     │
├─────────────────────────────────────────────┤
│ 📊 月間統計                                  │
│ ├─ 総フィードバック: 1,250 件               │
│ ├─ 平均精度: 84.2%                          │
│ ├─ 👍 率: 84%                               │
│ └─ 👎 率: 16%                               │
├─────────────────────────────────────────────┤
│ 📈 クエリ別精度ランキング                    │
│ ├─ 【高精度】飲食業リース: 95%              │
│ ├─ 【中精度】小売店舗: 82%                 │
│ └─ 【低精度】建設機械: 71%                 │
├─────────────────────────────────────────────┤
│ 🎯 Weights 提案                             │
│ ├─ 現在: semantic 0.60, bm25 0.40          │
│ ├─ 提案: semantic 0.65, bm25 0.35          │
│ │ （理由: BM25 スコアのノイズを削減）       │
│ └─ [✓ 適用]  [✗ 却下]  [? レビュー]       │
├─────────────────────────────────────────────┤
│ 🔴 低精度クエリ（要改善）                    │
│ ├─ 建設機械リース: 71% (n=47)              │
│ ├─ 医療機器リース: 74% (n=32)              │
│ └─ [詳細を見る...]                        │
└─────────────────────────────────────────────┘
```

---

## 💾 データ蓄積・スキーマ

### 1. フィードバックログ形式

**ファイル**: `mobile_app/feedback_log.jsonl`

```jsonl
{"timestamp": "2026-05-30T14:32:15Z", "user_id": "user_001", "query": "飲食業リース", "result_id": "doc_42", "result_title": "飲食業の赤字対応", "result_rank": 1, "feedback": "helpful", "search_type": "hybrid", "latency_ms": 145, "metadata": {"session_id": "sess_abc123", "device": "web"}}
{"timestamp": "2026-05-30T14:35:22Z", "user_id": "user_002", "query": "建設機械 リース", "result_id": "doc_15", "result_title": "建設機械の残価評価", "result_rank": 2, "feedback": "not_helpful", "search_type": "hybrid", "latency_ms": 203, "metadata": {"session_id": "sess_def456", "device": "mobile"}}
```

### 2. フィードバックデータスキーマ

```python
class FeedbackRecord:
    """ユーザーフィードバック記録"""
    
    # 識別情報
    timestamp: datetime          # UTC タイムスタンプ
    user_id: str                # ユーザー ID
    session_id: str             # セッション ID
    
    # クエリ情報
    query: str                  # 検索クエリ
    query_normalized: str       # 正規化されたクエリ
    
    # 検索結果情報
    result_id: str              # ドキュメント ID
    result_title: str           # ドキュメントタイトル
    result_rank: int            # 検索結果での順位
    result_score: float         # Hybrid Search スコア
    
    # フィードバック
    feedback: Literal["helpful", "not_helpful"]  # フィードバック内容
    
    # メタデータ
    search_type: str            # "hybrid" / "semantic" / "vector"
    latency_ms: float           # 検索レイテンシ
    metadata: Dict[str, Any]    # デバイス、ブラウザなど
```

### 3. 集計テーブル（SQLite）

**テーブル**: `feedback_metrics_daily`

```sql
CREATE TABLE feedback_metrics_daily (
    date DATE PRIMARY KEY,
    total_feedback INT,
    helpful_count INT,
    not_helpful_count INT,
    accuracy_rate FLOAT,
    avg_latency_ms FLOAT,
    query_count INT,
    unique_users INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE query_feedback_stats (
    query_normalized TEXT PRIMARY KEY,
    total_feedback INT,
    helpful_count INT,
    not_helpful_count INT,
    accuracy_rate FLOAT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📊 データ蓄積・分析ロジック

### 1. リアルタイムフィードバック収集

**クラス**: `FeedbackCollector`

```python
class FeedbackCollector:
    """ユーザーフィードバックをリアルタイムで収集"""
    
    def __init__(self, log_file: str = "feedback_log.jsonl"):
        self.log_file = log_file
        self.queue = Queue()  # スレッドセーフキュー
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def record_feedback(self, feedback_record: FeedbackRecord):
        """
        フィードバックを非同期で記録
        
        Args:
            feedback_record: FeedbackRecord インスタンス
        """
        self.queue.put(feedback_record)
    
    def _worker(self):
        """バックグラウンドワーカー（バッチ処理）"""
        batch = []
        while True:
            try:
                # キューからアイテムを取得（タイムアウト 5 秒）
                record = self.queue.get(timeout=5)
                batch.append(record)
                
                # バッチサイズ 100 に達したら書き込み
                if len(batch) >= 100:
                    self._flush_batch(batch)
                    batch = []
            except Empty:
                # タイムアウト時もバッチがあれば書き込み
                if batch:
                    self._flush_batch(batch)
                    batch = []
    
    def _flush_batch(self, batch: List[FeedbackRecord]):
        """バッチをファイルに書き込み"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            for record in batch:
                json_line = json.dumps(record.to_dict(), ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(f"💾 {len(batch)} 件のフィードバックを保存")
```

### 2. 日次分析・集計

**クラス**: `FeedbackAnalyzer`

```python
class FeedbackAnalyzer:
    """フィードバックの分析・統計"""
    
    def __init__(self, log_file: str, db_path: str):
        self.log_file = log_file
        self.db_path = db_path
    
    def analyze_daily(self, date: datetime = None) -> Dict[str, Any]:
        """
        日次分析（毎日 24:00 に実行予定）
        
        Returns:
            {
                "date": "2026-05-30",
                "total_feedback": 250,
                "helpful_count": 210,
                "not_helpful_count": 40,
                "accuracy_rate": 84.0,
                "avg_latency_ms": 142.5,
                "by_query": {
                    "飲食業リース": {"accuracy": 95, "count": 80},
                    "建設機械リース": {"accuracy": 71, "count": 47},
                    ...
                }
            }
        """
        # 当日のフィードバックを読み込む
        records = self._load_records_for_date(date or datetime.now())
        
        if not records:
            return {}
        
        # 集計
        helpful = sum(1 for r in records if r.feedback == "helpful")
        total = len(records)
        accuracy = (helpful / total * 100) if total > 0 else 0
        
        # クエリ別集計
        by_query = self._aggregate_by_query(records)
        
        # DB に保存
        self._save_metrics(date or datetime.now(), helpful, total, accuracy)
        
        return {
            "date": (date or datetime.now()).strftime("%Y-%m-%d"),
            "total_feedback": total,
            "helpful_count": helpful,
            "not_helpful_count": total - helpful,
            "accuracy_rate": round(accuracy, 1),
            "by_query": by_query
        }
    
    def _aggregate_by_query(self, records: List[FeedbackRecord]) -> Dict:
        """クエリ別の精度を集計"""
        query_stats = defaultdict(lambda: {"helpful": 0, "total": 0})
        
        for record in records:
            q = record.query_normalized
            query_stats[q]["total"] += 1
            if record.feedback == "helpful":
                query_stats[q]["helpful"] += 1
        
        return {
            q: {
                "accuracy": round(stats["helpful"] / stats["total"] * 100, 1),
                "count": stats["total"]
            }
            for q, stats in query_stats.items()
        }
```

### 3. Weights 動的調整提案

**クラス**: `WeightsOptimizer`

```python
class WeightsOptimizer:
    """Hybrid Search の Weights を動的に最適化"""
    
    def propose_new_weights(self, feedback_metrics: Dict) -> Dict:
        """
        フィードバック分析から新しい Weights を提案
        
        Logic:
        1. クエリ別精度が低いクエリを特定
        2. その原因（Semantic vs BM25）を推測
        3. Weights 調整を提案
        
        Returns:
            {
                "proposed_weights": {
                    "semantic_weight": 0.65,
                    "bm25_weight": 0.35
                },
                "reasoning": "...",
                "impact_estimate": {...}
            }
        """
        # 低精度クエリを特定（精度 < 80%）
        low_accuracy_queries = {
            q: metrics
            for q, metrics in feedback_metrics.get("by_query", {}).items()
            if metrics.get("accuracy", 100) < 80
        }
        
        if not low_accuracy_queries:
            return {"proposed_weights": None, "reasoning": "全クエリで高精度"}
        
        # 原因分析（ここでは簡略化）
        # - BM25 スコアが高い単語を含むクエリ → BM25 weight 低下
        # - セマンティックな意味が重要 → semantic weight 上昇
        
        reasoning = f"{len(low_accuracy_queries)} 件の低精度クエリを検出。"
        reasoning += "BM25 のノイズを削減するため semantic weight を上昇。"
        
        proposed = {
            "semantic_weight": min(0.70, 0.60 + len(low_accuracy_queries) * 0.01),
            "bm25_weight": 0.30
        }
        
        return {
            "proposed_weights": proposed,
            "reasoning": reasoning,
            "confidence": 0.7  # 提案の信頼度
        }
```

### 4. スケジュール・自動実行

**実行スケジュール** (cron):

```
# 日次分析（毎日 00:30）
30 0 * * * python mobile_app/rag_daily_maintenance.py analyze_feedback

# 週次レビュー（毎週月曜 10:00）
0 10 * * 1 python mobile_app/rag_daily_maintenance.py weekly_review

# 月次 Weights 自動提案（月初 1 日 9:00）
0 9 1 * * python mobile_app/rag_daily_maintenance.py propose_weights
```

---

## 🔄 実装フロー

### Phase 1: UI フィードバック機構（1 週間）

1. **フィードバック UI コンポーネント実装**
   - React コンポーネント: `FeedbackButtons.tsx`
   - スタイル: CSS Module
   - アニメーション: framer-motion

2. **API エンドポイント実装**
   ```
   POST /api/feedback
   {
       "query": "...",
       "result_id": "...",
       "feedback": "helpful" | "not_helpful"
   }
   ```

3. **チャット画面への統合**
   - 検索結果コンポーネントに フィードバック UI を追加

### Phase 2: データ蓄積機構（1 週間）

1. **FeedbackCollector 実装**
   - キュー + バックグラウンドワーカー
   - `feedback_log.jsonl` 追記

2. **DB テーブル作成**
   - `feedback_metrics_daily`
   - `query_feedback_stats`

3. **ロギング**
   - すべてのフィードバックを JSONL で永続化

### Phase 3: 分析・最適化機構（1 週間）

1. **FeedbackAnalyzer 実装**
   - 日次集計
   - クエリ別精度分析

2. **WeightsOptimizer 実装**
   - 自動提案ロジック
   - 管理者レビュー機構

3. **ダッシュボード実装**
   - 日次・週次・月次レポート
   - 可視化（Chart.js）

### Phase 4: 自動化・運用（1 週間）

1. **スケジュール実行設定**
   - cron ジョブ + APScheduler

2. **アラート機構**
   - 精度が急落した際にメール通知

3. **ドキュメント・トレーニング**

---

## 🔐 セキュリティ・プライバシー考慮

| 項目 | 対策 |
|------|------|
| **個人情報保護** | `user_id` はハッシュ化。実際のユーザー名は記録しない |
| **ログローテーション** | `feedback_log.jsonl` を 1MB ごとにローテーション |
| **アクセス制限** | ダッシュボードは管理者のみ（認証・認可） |
| **データ保持期限** | 12 ヶ月以上のフィードバックはアーカイブ |

---

## 📈 期待効果・指標

### 短期（3 ヶ月）

- フィードバック蓄積: 1,000 件/月
- 検索精度: 95% → 96%（+1 pt）
- 低精度クエリの特定

### 中期（6 ヶ月）

- Weights 自動調整の実運用開始
- 検索精度: 96% → 98%（+2 pt）
- クエリ別カスタマイズの検討

### 長期（1 年）

- 検索精度: 98% を超える
- ユーザー満足度向上
- 業種別・リース種別別の最適化

---

## ⚠️ リスク・対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| **フィードバック量不足** | 中 | 分析精度低下 | ユーザー教育・UI 改善 |
| **Weights 自動調整失敗** | 低 | 精度低下 | 管理者レビュー化 |
| **ログファイル肥大化** | 低 | ディスク満杯 | ローテーション・圧縮 |
| **ユーザー不信感** | 低 | 利用離脱 | プライバシーポリシー明示 |

---

## 📊 見積もり・スケジュール

| タスク | 工数 | 期間 | 開始時期 |
|--------|------|------|---------|
| **Phase 1: UI** | 40h | 1 週 | PHASE 2 完了後 +1 週 |
| **Phase 2: データ蓄積** | 30h | 1 週 | +2 週 |
| **Phase 3: 分析・最適化** | 35h | 1 週 | +3 週 |
| **Phase 4: 自動化・運用** | 20h | 1 週 | +4 週 |
| **テスト・ドキュメント** | 25h | 1 週 | 並行 |

**総工数**: 150h ≒ 3-4 週間（1 人日 8h, 土日除く）

---

## 🎯 成功基準

- [x] フィードバック UI が正常に動作
- [x] 1 週目で最低 100 件のフィードバック蓄積
- [x] 日次分析が毎日実行される
- [x] Weights 提案が月 1 回以上実施
- [x] 検索精度が 1% 以上向上

---

## 📝 備考・今後の検討事項

1. **多言語対応**: 現在は日本語のみ。英語・中国語への拡張を検討
2. **A/B テスト**: Weights 提案を複数案で検証
3. **機械学習**: Feedback → Weights の自動最適化（scikit-learn）
4. **ユーザーセグメント**: 業種別・企業規模別の独立した Weights 管理
5. **フィードバック理由の拡張**: 「役に立たない」理由の詳細化（複数選択肢）

---

**作成日**: 2026-05-30  
**ステータス**: 企画案  
**レビュー待ち**: —

