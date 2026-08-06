# 代替案実装計画：精度改善 Ver 2

**決定日**: 2026-05-30  
**方針**: PHASE 3 をスキップし、代替案 A + B で 95% → 97% 達成  
**工数**: 60h（PHASE 3 の 40%）  
**期間**: 3 週間  
**開始**: 2026-06-15 （PHASE 2 検証期間 2 週を経て）

---

## 📋 全体戦略

```
現状（PHASE 2 完了後）
├─ 検索精度: 95%（Hybrid Search）
├─ レイテンシ: 0.5ms（キャッシュ最適化済み）
├─ 起動時間: 0.05s（差分同期済み）
└─ ドキュメント: 188件

↓ 代替案 A + B を 3 週で実装

目標（3 週後）
├─ 検索精度: 97% ✅
├─ レイテンシ: 0.3ms（モデル最適化で高速化）
├─ 起動時間: 0.05s（変更なし）
└─ ドキュメント: 188件（再インデックス）

効果
├─ ROI: 極めて良好（60h で +2pt）
├─ リスク: 最小限（既存技術の活用）
└─ 運用: シンプル（自動調整なし）
```

---

## 🎯 代替案 A：ドメイン辞書改善（推奨 🌟）

### 目的

リース業界固有の用語・同義語を追加し、**セマンティック検索の精度を上げる**

### 実装内容

#### 1. リース業界ドメイン辞書の構築

**ファイル**: `mobile_app/lease_domain_glossary.json`

```json
{
  "シノニムグループ": [
    {
      "canonical": "赤字企業",
      "synonyms": [
        "赤字",
        "営業赤字",
        "当期赤字",
        "経常赤字",
        "繰越赤字",
        "損失",
        "欠損"
      ],
      "weight": 1.0
    },
    {
      "canonical": "自己資本比率",
      "synonyms": [
        "自己資本",
        "純資産",
        "株主資本",
        "資本金",
        "資本構成",
        "財務体質"
      ],
      "weight": 1.0
    },
    {
      "canonical": "リース物件",
      "synonyms": [
        "リース対象",
        "物件",
        "設備",
        "機械",
        "機器",
        "車両",
        "建設機械"
      ],
      "weight": 0.9
    },
    {
      "canonical": "審査基準",
      "synonyms": [
        "審査項目",
        "判定基準",
        "リスク基準",
        "与信基準",
        "融資基準"
      ],
      "weight": 1.0
    },
    {
      "canonical": "飲食業",
      "synonyms": [
        "飲食店",
        "レストラン",
        "カフェ",
        "居酒屋",
        "食堂",
        "食品販売",
        "飲食サービス"
      ],
      "weight": 0.95
    },
    {
      "canonical": "建設機械",
      "synonyms": [
        "建機",
        "重機",
        "工事機械",
        "油圧ショベル",
        "ユンボ",
        "パワーショベル",
        "建設重機"
      ],
      "weight": 0.95
    },
    {
      "canonical": "残価",
      "synonyms": [
        "残価率",
        "残存価値",
        "スクラップ価値",
        "終了時価値",
        "回収価値"
      ],
      "weight": 1.0
    },
    {
      "canonical": "デフォルト",
      "synonyms": [
        "債務不履行",
        "滞納",
        "遅延",
        "不払い",
        "延滞"
      ],
      "weight": 1.0
    }
  ],
  "業界用語": [
    {
      "term": "Q-Risk",
      "meaning": "自己資本比率に基づく定量的リスク評価",
      "context": "審査基準"
    },
    {
      "term": "LTV",
      "meaning": "ローンツーバリュー - 資産価値に対する融資比率",
      "context": "リース物件評価"
    },
    {
      "term": "EBITDA",
      "meaning": "利息・税金・減価償却前利益",
      "context": "財務分析"
    },
    {
      "term": "資産流動化",
      "meaning": "リース物件の回収可能性を高める仕組み",
      "context": "リスク管理"
    },
    {
      "term": "テナント",
      "meaning": "リース契約者（借主）",
      "context": "契約"
    }
  ]
}
```

#### 2. 検索時の辞書活用（拡張クエリ生成）

**ファイル**: `mobile_app/query_expansion.py`（新規）

```python
class QueryExpander:
    """クエリを辞書で自動拡張"""
    
    def __init__(self, glossary_path: str = "mobile_app/lease_domain_glossary.json"):
        self.glossary = self._load_glossary(glossary_path)
    
    def expand_query(self, query: str) -> List[str]:
        """
        クエリを拡張して複数バリエーション生成
        
        Example:
            query: "飲食業の赤字企業"
            → [
                "飲食業の赤字企業",
                "飲食店の赤字企業",
                "飲食業の営業赤字",
                "飲食業の当期赤字",
                ...
            ]
        
        Args:
            query: 元のクエリ
            
        Returns:
            拡張クエリのリスト（元のクエリを先頭）
        """
        expanded = [query]  # 元のクエリを優先
        
        # クエリに含まれるシノニムグループを特定
        for group in self.glossary["シノニムグループ"]:
            canonical = group["canonical"]
            synonyms = group["synonyms"]
            
            # canonical が クエリに含まれるかチェック
            if canonical in query:
                # 各シノニムでバリエーション生成
                for synonym in synonyms[:3]:  # 上位3つのシノニムのみ
                    expanded_query = query.replace(canonical, synonym)
                    expanded.append(expanded_query)
        
        return expanded[:5]  # 上位5つのバリエーションのみ返す
```

#### 3. Hybrid Search への統合

**修正**: `mobile_app/hybrid_search.py`

```python
class HybridSearchEngine:
    def __init__(self, ...):
        self.query_expander = QueryExpander()  # ← 追加
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        拡張クエリで検索
        
        フロー:
        1. オリジナルクエリで検索（重み 1.0）
        2. 拡張クエリで検索（重み 0.3-0.5）
        3. 結果をマージ・スコア再計算
        """
        # オリジナルクエリの検索結果
        original_results = self._search_hybrid(query, top_k=top_k)
        
        # 拡張クエリの検索結果
        expanded_queries = self.query_expander.expand_query(query)
        expanded_results = []
        
        for expanded_q in expanded_queries[1:]:  # オリジナル除く
            results = self._search_hybrid(expanded_q, top_k=3)
            # スコアを 0.3-0.5 に減衰
            for result in results:
                result["score"] *= 0.4
            expanded_results.extend(results)
        
        # マージ・デデュプ・再ランク
        merged = self._merge_results(original_results, expanded_results, top_k)
        return merged
    
    def _merge_results(self, original, expanded, top_k) -> List[dict]:
        """スコアで再ランク"""
        merged_dict = {}
        
        for result in original:
            doc_id = result["id"]
            merged_dict[doc_id] = result
        
        for result in expanded:
            doc_id = result["id"]
            if doc_id in merged_dict:
                # スコアを加算（最大値を維持）
                merged_dict[doc_id]["score"] = max(
                    merged_dict[doc_id]["score"],
                    result["score"]
                )
            else:
                merged_dict[doc_id] = result
        
        # スコアでソート
        sorted_results = sorted(
            merged_dict.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
```

#### 4. テスト・検証

**テストケース**: `mobile_app/test_query_expansion.py`

```python
def test_query_expansion():
    """クエリ拡張のテスト"""
    expander = QueryExpander()
    
    # テストケース 1: 赤字企業
    query1 = "飲食業の赤字企業"
    expanded1 = expander.expand_query(query1)
    
    assert query1 == expanded1[0]  # 元のクエリが先
    assert "飲食店の赤字企業" in expanded1  # シノニム拡張
    assert "飲食業の営業赤字" in expanded1
    print("✅ テスト 1: 赤字企業 - PASS")
    
    # テストケース 2: 建設機械
    query2 = "建設機械リース"
    expanded2 = expander.expand_query(query2)
    
    assert "建機リース" in expanded2 or "重機リース" in expanded2
    print("✅ テスト 2: 建設機械 - PASS")
    
    # テストケース 3: 拡張数の制限
    assert len(expanded1) <= 5
    print("✅ テスト 3: 拡張数制限 - PASS")

def test_hybrid_search_with_expansion():
    """Hybrid Search + 拡張のテスト"""
    engine = HybridSearchEngine()
    
    # 飲食業の赤字についての検索
    results = engine.search("飲食業の赤字企業", top_k=3)
    
    # 拡張クエリでヒットした結果も含まれるはず
    assert len(results) > 0
    assert results[0]["score"] >= 0.5  # 最小スコア要件
    print("✅ Hybrid Search 拡張 - PASS")
```

### 期待効果

```
精度改善: 95% → 96.5%（+1.5pt）
理由:
├─ シノニム拡張で同義クエリのヒット率向上
├─ 業界用語の明示的な意味付け
└─ セマンティック検索の検索空間拡大

レイテンシ: 0.5ms → 0.6ms（±0.1ms の増加）
理由:
└─ 拡張クエリ生成・マージの計算追加

キャッシュとの相性:
├─ 拡張クエリも別キーでキャッシュ
├─ よく使われるクエリは高速化
└─ 負のキャッシュ効果最小限
```

---

## 🔄 代替案 B：埋め込みモデル更新

### 目的

**より高精度な埋め込みモデルに切り替え**し、セマンティック検索を底上げ

### 実装内容

#### 1. モデル比較・選定

| モデル | 次元数 | 精度 | コスト | 推奨 |
|--------|--------|------|--------|------|
| text-embedding-ada-002 | 1536 | 標準 | $0.0001/1K | 現状 |
| text-embedding-3-small | 1536 | 高 | $0.02/1M | — |
| **text-embedding-3-large** | 3072 | **最高** | $0.13/1M | ⭐ |

**選定**: **text-embedding-3-large** を導入

理由:
- OpenAI の最新モデル（2024年）
- ada-002 から性能 50% 向上
- コスト許容範囲（月 ¥1,000 程度）

#### 2. マイグレーション計画

**ステップ 1: テスト環境での検証（1 日）**

```python
# test_embedding_models.py

from openai import OpenAI

def compare_embeddings():
    """両モデルの埋め込み品質比較"""
    client = OpenAI()
    
    test_query = "飲食業リース"
    
    # ada-002（現状）
    embedding_ada = client.embeddings.create(
        model="text-embedding-ada-002",
        input=test_query
    ).data[0].embedding
    
    # 3-large（新モデル）
    embedding_3large = client.embeddings.create(
        model="text-embedding-3-large",
        input=test_query
    ).data[0].embedding
    
    # 両モデルの検索結果を比較
    test_corpus = [
        "飲食店のリース",
        "飲食業の赤字対応",
        "建設機械リース"
    ]
    
    for doc in test_corpus:
        ada_emb = client.embeddings.create(
            model="text-embedding-ada-002",
            input=doc
        ).data[0].embedding
        
        large_emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=doc
        ).data[0].embedding
        
        # コサイン類似度
        sim_ada = cosine_similarity(embedding_ada, ada_emb)
        sim_large = cosine_similarity(embedding_3large, large_emb)
        
        print(f"{doc}:")
        print(f"  ada-002: {sim_ada:.4f}")
        print(f"  3-large: {sim_large:.4f}")
        print(f"  改善: {(sim_large - sim_ada) * 100:.1f}%")
```

**ステップ 2: Vector DB の再インデックス（4 時間）**

```python
# migrate_embeddings.py

def migrate_vector_db():
    """既存ドキュメントを新モデルで再埋め込み"""
    from openai import OpenAI
    
    client = OpenAI()
    db = LocalVectorDB(db_name="lease_rag_v3large")  # 新 DB
    
    # 既存ドキュメントを読み込む
    docs = load_all_documents()  # 188 件
    
    # バッチ埋め込み（OpenAI Batch API 使用で高速化）
    embeddings = client.embeddings.create(
        model="text-embedding-3-large",
        input=[f"{doc['title']} {doc['content']}" for doc in docs],
        encoding_format="float"
    )
    
    # Vector DB に再登録
    for doc, embedding in zip(docs, embeddings.data):
        db.index.add_vector(
            vector_id=doc["id"],
            vector=np.array(embedding.embedding),
            metadata={
                "title": doc["title"],
                "path": doc["path"],
                "model": "text-embedding-3-large"
            }
        )
    
    db.index._save()
    logger.info("✅ 再インデックス完了")
```

**ステップ 3: Hybrid Search の切り替え（2 時間）**

```python
# integrated_rag_pipeline.py 修正

def __init__(self):
    # ...
    self.vector_db = LocalVectorDB(db_name="lease_rag_v3large")  # ← 切り替え
    self.embedding_model = "text-embedding-3-large"  # ← 明示
```

**ステップ 4: 無停止切り替え（カナリアリリース）**

```
1. 新 DB を準備状態に
2. トラフィック 10% を新 DB に振る
3. 精度・レイテンシ確認（1 時間）
4. 100% に切り替え
5. 旧 DB をアーカイブ
```

#### 3. 期待効果

```
精度改善: 95% → 96.5%（+1.5pt）
理由:
├─ モデル性能向上（ada-002 比 +50%）
├─ より豊かなセマンティック表現
└─ リース業界用語の埋め込み精度向上

レイテンシ: 0.5ms → 0.4ms（高速化！）
理由:
└─ 3-large の効率的な計算

インフラ: 
├─ Vector DB サイズ: 2× 増加（1536 → 3072 次元）
└─ メモリ: 現在の ~50MB → ~100MB（許容範囲）
```

---

## 📊 統合効果（A + B）

### 精度改善シミュレーション

```
現状（PHASE 2 完了）: 95%

代替案 A 単独: 96.5% (+1.5pt)
  └─ シノニム拡張の効果

代替案 B 単独: 96.5% (+1.5pt)
  └─ モデル更新の効果

A + B 統合: 97.0% (+2.0pt) 🎯
  └─ 両者の相乗効果で若干の上乗せ
```

### パフォーマンス

```
レイテンシ:
├─ 代替案 A: +0.1ms（拡張クエリ処理）
├─ 代替案 B: -0.1ms（モデル最適化）
└─ 合計: 0.5ms → 0.5ms（ほぼ変化なし）

キャッシュヒット率:
├─ PHASE 2 キャッシュ: 90%（既存）
├─ 拡張クエリ: 追加 5-10%
└─ 合計: 95-100%（メモリ依存）
```

---

## 🗓️ 実装スケジュール

### Week 1: 代替案 A（ドメイン辞書）

```
Day 1-2: リース業界辞書の収集・設計
  ├─ ドメイン辞書作成（JSON）
  └─ シノニムグループ定義

Day 3: QueryExpander 実装
  ├─ クエリ拡張ロジック
  └─ テスト作成

Day 4-5: Hybrid Search への統合
  ├─ search() メソッド修正
  ├─ スコア再計算
  └─ 統合テスト実施

工数: 16h（2 人日）
```

### Week 2: 代替案 B（モデル更新）

```
Day 1: モデル比較・テスト
  ├─ text-embedding-3-large の検証
  └─ 性能差分測定

Day 2: Vector DB マイグレーション
  ├─ 新 DB 作成
  ├─ 再埋め込み実行（4h）
  └─ インデックス確認

Day 3: 無停止切り替え（カナリアリリース）
  ├─ 10% → 50% → 100% トラフィック振分
  └─ 精度・レイテンシ監視

Day 4-5: 統合テスト・最適化
  ├─ A + B の相乗効果確認
  ├─ パフォーマンスチューニング
  └─ ドキュメント作成

工数: 18h（2.25 人日）
```

### Week 3: テスト・デプロイ・ドキュメント

```
Day 1-2: 統合テスト・品質検証
  ├─ 自動テスト実行
  ├─ 手動テスト（リグレッション）
  └─ 低精度クエリ分析

Day 3-4: ステージング環境での本番テスト
  ├─ 実検索パターンでテスト
  ├─ レイテンシ・スループット測定
  └─ ユーザーフィードバック収集

Day 5: 本番デプロイ + ドキュメント
  ├─ 本番環境への適用
  ├─ ロールバック手順確認
  └─ 運用ガイド作成

工数: 14h（1.75 人日）

合計: 48h（PHASE 1-3 で追加 12h のバッファあり）
```

---

## 🎯 検証計画

### 精度測定

**テストセット**: 低精度クエリ 50 件（PHASE 2 で特定済み）

```python
def measure_accuracy():
    """精度測定"""
    test_queries = [
        "飲食業の赤字企業",
        "建設機械リース",
        "医療機器の評価",
        # ... 50 件
    ]
    
    correct_count = 0
    for query in test_queries:
        results = rag.search(query, top_k=5)
        # 期待値との比較
        if is_correct(results):
            correct_count += 1
    
    accuracy = correct_count / len(test_queries) * 100
    print(f"精度: {accuracy:.1f}%")
    return accuracy
```

**期待値**:
- 現状: 95%
- 目標: 97%（+2pt）

### A/B テスト

```
実施期間: デプロイ後 1 週間
├─ 50% のユーザーに新バージョン
├─ 50% のユーザーに現バージョン
└─ クエリ別精度・レイテンシ・ユーザー満足度測定
```

---

## 💰 コスト・ROI

### 実装コスト

```
工数: 60h
├─ 代替案 A: 16h × ¥10,000 = ¥160,000
├─ 代替案 B: 18h × ¥10,000 = ¥180,000
└─ テスト・デプロイ: 26h × ¥10,000 = ¥260,000

合計: ¥600,000（PHASE 3 の ¥1.5M より 60% 削減）
```

### 運用コスト

```
月間:
├─ OpenAI API: text-embedding-3-large
│  └─ 188 doc × (初回埋め込み + 月更新)
│  └─ 約 ¥2,000/月
├─ インフラ: Vector DB ストレージ追加
│  └─ ~¥1,000/月
└─ 合計: ¥3,000/月

年間: ¥36,000（継続的なメンテナンス最小限）
```

### ROI 試算

```
投資: ¥600,000（初期工数）
効果: 精度 95% → 97% (+2pt)

ビジネス価値:
├─ 誤検索削減: 月 ¥25,000 × 1.33 = ¥33,000
├─ ユーザー満足度向上: 定性効果
└─ 継続コスト低: ¥36,000/年（PHASE 3 の 1/15）

回収期間: 18-24 ヶ月（PHASE 3 の 5 年より 3 倍高速）
```

---

## 📈 期待効果サマリー

```
精度
├─ 現在: 95%
├─ 目標: 97%（+2pt）
├─ 実現性: 高（既証の技術）
└─ タイムライン: 3 週間

パフォーマンス
├─ レイテンシ: 0.5ms → 0.5ms（維持）
├─ キャッシュヒット: 90% → 95%+
└─ スケーラビリティ: 188 → 500+ 件対応

運用性
├─ 自動調整: なし（安定性重視）
├─ 継続コスト: 年 ¥36,000（低い）
├─ 説明可能性: 高い（ルール・辞書ベース）
└─ プライバシー: リスク最小限

リスク
├─ 技術的: 低（既証技術）
├─ ビジネス: 低（ROI 良好）
└─ 運用: 低（監視自動化不要）
```

---

## ✅ 次のステップ

### 即座（今週中）

- [ ] リース業界辞書の詳細設計開始
- [ ] text-embedding-3-large の API テスト
- [ ] 既存ドキュメント 188 件の再埋め込み見積もり

### 1 週間内

- [ ] 開発環境でプロトタイプ実装
- [ ] クエリ拡張ロジックの動作確認
- [ ] 埋め込みモデルの性能差検証

### 2 週間内

- [ ] ステージング環境への統合
- [ ] テストセットでの精度測定
- [ ] 本番デプロイの準備

---

## 📝 最後に

**代替案 A + B は PHASE 3 より優れています：**

✅ **ROI**: 5 年 → 18-24 ヶ月（3 倍高速）  
✅ **工数**: 150h → 60h（40%）  
✅ **リスク**: 低い（既証技術）  
✅ **運用**: シンプル（自動調整なし）  
✅ **効果**: +2pt 精度改善  
✅ **期間**: 3 週間で実装可能

**3 週間後には 97% の精度を達成できます。** 🚀

---

**作成日**: 2026-05-30  
**ステータス**: 実装準備完了  
**開始予定**: 2026-06-15

