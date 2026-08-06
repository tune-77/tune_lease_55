# AI Chat 向け RAG 改善計画

**目的**: AI Chat ユーザーとのインタラクション中に RAG を最大限活用し、回答品質を向上させる

**前提**: ユーザーが質問 → RAG が知識を検索 → AI が統合して回答

---

## 🎯 8つの改善施策

### 1️⃣ リアルタイム RAG 統合（会話スレッド内検索）

**現状**:
- チャット前に一度だけ検索
- 会話が進んでも同じ RAG 結果

**改善**:
```python
# ユーザーメッセージごとに RAG 検索を更新
async def chat_with_dynamic_rag(user_message: str, conversation_history):
    # 現在のコンテキスト + ユーザー質問から動的に RAG 検索
    rag_results = search_with_wikilink_context(
        query=user_message,  # 最新の質問で検索
        limit=5
    )
    
    # AI が回答生成時に最新の RAG 結果を使用
    response = await llm.generate(
        prompt=build_prompt_with_rag(
            conversation_history,
            user_message,
            rag_results  # ← 毎メッセージ更新
        )
    )
    
    return response, rag_results  # 結果も返す
```

**効果**:
- 会話の進展に応じて RAG 結果が自動更新
- 「前のメッセージでは見つからなかった情報」も後で発見可能
- より正確で新鮮な情報に基づいた回答

**実装優先度**: 🔴 高（即効性大）

---

### 2️⃣ RAG 結果のコンテキスト圧縮（トークン効率化）

**現状**:
```
ユーザー質問 (50 tokens)
  ↓
RAG 検索結果 (1000+ tokens)
  ↓
LLM が処理（トークン数オーバー警告）
```

**改善**:
```python
class RagContextCompressor:
    """RAG 結果をコンパクト化。"""
    
    @staticmethod
    def compress_results(rag_results: list[dict]) -> str:
        """関連部分のみを抽出・圧縮。"""
        compressed = []
        
        for result in rag_results:
            # 1. スニペットから最関連部分を抽出
            key_sentences = extract_key_sentences(
                result["snippet"],
                limit=2  # 最重要文のみ
            )
            
            # 2. 重複排除
            if key_sentences not in compressed:
                compressed.append({
                    "source": result["path"],
                    "content": key_sentences,
                    "relevance": result.get("score", 0.5),
                })
        
        # 3. Markdown テーブル形式に圧縮
        table = "| ソース | 内容 | 関連度 |\n|--------|------|-----|\n"
        for item in compressed[:3]:  # TOP 3 のみ
            table += f"| {item['source']} | {item['content']} | {item['relevance']:.1f} |\n"
        
        return table
```

**効果**:
- RAG 結果: 1000 tokens → 200 tokens （80%削減）
- 同じ品質で LLM コスト 削減
- より長い会話履歴を保持可能

**実装優先度**: 🟡 中（コスト最適化）

---

### 3️⃣ 会話履歴の RAG 統合（コンテキスト学習）

**現状**:
- 各質問は独立（過去のチャットから学ばない）
- ユーザーが繰り返し同じ質問をする

**改善**:
```python
async def chat_with_conversation_memory(user_message: str):
    """過去のチャットから RAG 検索クエリを拡張。"""
    
    # Phase 1: 会話履歴から文脈を抽出
    conversation_context = extract_context_from_history(
        conversation_history[-5:]  # 最新5メッセージ
    )
    
    # Phase 2: 質問 + 会話文脈を組み合わせて RAG 検索
    expanded_query = f"{user_message} + {conversation_context}"
    rag_results = search_notes(expanded_query, limit=5)
    
    # Phase 3: 会話履歴内で既出の情報は優先度下げ
    for result in rag_results:
        if any(result["snippet"] in msg for msg in conversation_history):
            result["score"] *= 0.5  # 既出情報は減点
    
    return sorted(rag_results, key=lambda x: -x["score"])

# 例：
# ユーザー: 「製造業の審査について教えて」
# AI: 「製造業は...」
# ユーザー: 「Q-Risk について」← 自動的に「製造業 + Q-Risk」で検索
```

**効果**:
- ユーザーが「前の話題に関連して」と言わなくても AI が理解
- より正確な RAG 検索が自動実行
- 会話がより自然で連続性がある

**実装優先度**: 🟡 中（UX 向上）

---

### 4️⃣ RAG 信頼度スコア表示（透明性向上）

**現状**:
```
AI: 「Q-Risk が高い場合、追加調査が必要です」
↓ ユーザーは「どこから来た情報？」と疑問
```

**改善**:
```python
async def chat_with_confidence_display(user_message: str):
    """RAG 結果の信頼度をスコア表示。"""
    
    rag_results = search_with_wikilink_context(user_message)
    
    # 信頼度スコアを計算
    confidence_scores = {}
    for result in rag_results:
        score = (
            result.get("score", 0.5) * 0.4 +      # 関連度スコア
            calculate_source_reliability(result["path"]) * 0.3 +  # ソース信頼度
            calculate_recency(result["metadata"]["modified_at"]) * 0.3  # 新鮮度
        )
        confidence_scores[result["path"]] = score
    
    # AI 回答に信頼度バッジを付与
    response = await llm.generate(prompt_with_rag(rag_results))
    
    # 回答にメタデータ付与
    response_with_metadata = {
        "text": response,
        "sources": [
            {
                "path": r["path"],
                "confidence": confidence_scores[r["path"]],
                "badge": "🟢 高信頼" if confidence_scores[r["path"]] > 0.8 else "🟡 中",
            }
            for r in rag_results[:3]
        ]
    }
    
    return response_with_metadata
```

**UI 例**:
```
【AI 回答】
Q-Risk が高い場合、以下の点を確認してください...

📚 参考ノート:
🟢 高信頼 | Projects/tune_lease_55/Asset Knowledge/物件ファイナンス.md
🟡 中信頼 | AI Chat/2026-05-26 (3日前)
```

**効果**:
- ユーザーが「どこから来た情報か」を一目で理解
- 回答の信頼度が可視化
- ユーザーが必要に応じてソースを確認可能

**実装優先度**: 🟡 中（信頼性向上）

---

### 5️⃣ マルチモーダル RAG（画像・テーブル対応）

**現状**:
- テキストのみ検索
- Asset Knowledge の画像は検索不可

**改善**:
```python
class MultimodalRagSearch:
    """画像・テーブルも検索対象に。"""
    
    @staticmethod
    def search_all_modalities(query: str) -> list[dict]:
        """テキスト + 画像 + テーブルで検索。"""
        
        # 1. テキスト検索（既存）
        text_results = search_notes(query, limit=3)
        
        # 2. 画像検索（OCR + 埋め込み）
        image_results = []
        for image_file in find_images_in_vault():
            # 画像から OCR でテキスト抽出
            image_text = ocr_extract(image_file)
            if query_matches(query, image_text):
                image_results.append({
                    "type": "image",
                    "path": image_file,
                    "content": image_text,
                    "score": calculate_similarity(query, image_text),
                })
        
        # 3. テーブル検索（CSV/Markdown テーブル）
        table_results = []
        for table in find_tables_in_vault():
            if query_matches_table(query, table):
                table_results.append({
                    "type": "table",
                    "path": table["source"],
                    "content": render_table_as_markdown(table),
                    "score": calculate_relevance(query, table),
                })
        
        # 統合
        all_results = (
            text_results +
            sorted(image_results, key=lambda x: -x["score"])[:2] +
            sorted(table_results, key=lambda x: -x["score"])[:2]
        )
        
        return all_results[:5]  # TOP 5
```

**例**:
```
ユーザー: 「フォークリフトの残価推移を見せて」

【RAG 検索結果】
📄 テキスト: フォークリフト残価ガイド...
📊 テーブル: | 年式 | 評価額 | 残価率 | 
           | 新車 | ¥500万 | 100% |
           | 1年 | ¥350万 | 70% |
📷 画像: 中古フォークリフト相場表（2026年5月）
```

**実装優先度**: 🟢 低（高度な機能）

---

### 6️⃣ 動的プロンプト生成（RAG 結果に応じた最適化）

**現状**:
```
固定プロンプト: 
「あなたはリース審査専門家です。以下の情報に基づいて...」
```

**改善**:
```python
def generate_dynamic_prompt(
    user_message: str,
    rag_results: list[dict],
    conversation_history: list[dict]
) -> str:
    """RAG 結果の性質に応じてプロンプトを動的生成。"""
    
    # 1. RAG 結果から「必要な役割」を推定
    if any("金利" in r["snippet"] for r in rag_results):
        role = "金利推定の専門家"
    elif any("Q-Risk" in r["snippet"] for r in rag_results):
        role = "リスク分析の専門家"
    elif any("物件" in r["snippet"] for r in rag_results):
        role = "物件評価の専門家"
    else:
        role = "リース審査の専門家"
    
    # 2. RAG 結果から「参照スタイル」を決定
    if len(rag_results) > 3:
        reference_instruction = "以下の複数のノートから総合的に判断してください"
    else:
        reference_instruction = "以下のノートに基づいて詳しく説明してください"
    
    # 3. 会話文脈から「トーン」を調整
    if conversation_history[-1]["role"] == "user" and "簡潔に" in conversation_history[-1]["content"]:
        tone = "簡潔で要点のみ"
    else:
        tone = "詳細で根拠を含めて"
    
    # 4. 動的プロンプト生成
    prompt = f"""
あなたは{role}です。

{reference_instruction}:
{format_rag_results(rag_results)}

ユーザーの質問に対して、{tone}で回答してください。
回答時に、参照したノートを明示してください。

ユーザー質問: {user_message}
"""
    
    return prompt
```

**効果**:
- RAG 結果に応じて「最適な AI 役割」が自動選択
- より正確で、コンテキストに合った回答が生成
- AI が「どのノートを参考にしたか」を明確に述べる

**実装優先度**: 🟡 中（回答品質向上）

---

### 7️⃣ ユーザーフィードバックの RAG への反映

**現状**:
```
AI 回答を生成 → ユーザーが「参考になった / ならなかった」 → 終わり
```

**改善**:
```python
class FeedbackIntegratedRag:
    """ユーザーフィードバックを RAG 学習に反映。"""
    
    @staticmethod
    async def collect_feedback(response_id: str, feedback: str):
        """ユーザーフィードバックを記録。"""
        
        feedback_entry = {
            "response_id": response_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,  # "helpful" / "not helpful" / comment
            "used_rag_sources": get_sources_from_response(response_id),
        }
        
        # フィードバックを記録
        feedback_log = Path.home() / "Library" / "Logs" / "tunelease" / "rag_feedback.jsonl"
        feedback_log.write_text(
            feedback_log.read_text() +
            json.dumps(feedback_entry, ensure_ascii=False) + "\n"
        )
        
        # フィードバックに基づいて RAG を最適化
        if feedback == "helpful":
            # 役に立ったノートのスコアを上げる
            for source in feedback_entry["used_rag_sources"]:
                increment_source_score(source, +0.1)
        else:
            # 役に立たなかったノートのスコアを下げる
            for source in feedback_entry["used_rag_sources"]:
                increment_source_score(source, -0.05)

# UI 例:
# AI 回答の下に:
# 👍 この回答は役に立った | 👎 この回答は役に立たなかった | 💬 コメント
```

**効果**:
- ユーザーフィードバックが RAG スコアに反映
- 「実際に役に立つ情報」が自動的に上位にランクされる
- 継続的な改善ループが実現

**実装優先度**: 🟡 中（長期的改善）

---

### 8️⃣ AI Chat から見つかった Gap の改善候補化

**現状**:
```
ユーザー: 「この質問に対して RAG から情報が出ませんでした」
→ 誰も気づかない、改善候補にならない
```

**改善**:
```python
class GapDetectionSystem:
    """AI Chat から見つかった知識 Gap を自動検出。"""
    
    @staticmethod
    async def detect_and_report_gaps(
        user_query: str,
        rag_results: list[dict],
        ai_response: str
    ):
        """RAG が回答できなかったギャップを検出。"""
        
        # 1. ユーザー質問の意図を抽出
        intent = extract_intent(user_query)  # "金利計算", "Q-Risk評価" など
        
        # 2. RAG 検索の成功度を評価
        coverage = calculate_rag_coverage(rag_results, intent)
        
        # 3. ギャップがあれば改善候補化
        if coverage < 0.7:  # 70%未満なら Gap 検出
            gap = {
                "id": generate_id("GAP"),
                "type": "rag_gap",
                "topic": intent,
                "user_query": user_query,
                "missing_info": identify_missing_info(user_query, rag_results),
                "suggested_action": f"[[{intent}]] に関する Vault ノートを作成 / 拡充",
                "priority": "medium" if coverage > 0.5 else "high",
                "discovered_at": datetime.now().isoformat(),
                "chat_context": {
                    "user_message": user_query,
                    "ai_response": ai_response[:200],
                    "rag_results_count": len(rag_results),
                }
            }
            
            # 改善候補キューに追加
            add_to_improvement_queue(gap, category="rag_gap")
            
            # ユーザーに通知（オプション）
            notify_user(
                f"💡 この質問に対してもっと詳しい情報が必要かもしれません。"
                f"フィードバック: {gap['suggested_action']}"
            )

# 例：
# ユーザー: 「キャッシュフロー分析の具体的な方法は？」
# RAG が不十分だと検出
#   ↓
# 改善候補: "キャッシュフロー分析ガイドの作成" → dispatch_queue に追加
#   ↓
# 翌朝の朝見直しで優先度付け → 実装
```

**効果**:
- ユーザーの「質問したけど回答できなかった」が自動的に改善候補になる
- Vault が自動的に「ユーザーが必要とする知識」に進化
- AI Chat と Vault が相互に補完し合う

**実装優先度**: 🔴 高（継続改善の源泉）

---

## 📋 実装ロードマップ

### Phase 1: 基盤強化（1-2週間）
- [ ] 1️⃣ リアルタイム RAG 統合
- [ ] 8️⃣ Gap 検出システム

**効果**: 
- AI Chat の回答がリアルタイムに最新の知識を反映
- Vault が自動進化し始める

### Phase 2: ユーザー体験向上（2-3週間）
- [ ] 2️⃣ コンテキスト圧縮
- [ ] 4️⃣ 信頼度スコア表示

**効果**:
- 回答の透明性向上
- コスト削減

### Phase 3: 高度な機能（3-4週間）
- [ ] 3️⃣ 会話履歴の RAG 統合
- [ ] 6️⃣ 動的プロンプト生成

**効果**:
- AI Chat が「より自然で正確」に

### Phase 4: 継続改善（継続）
- [ ] 5️⃣ マルチモーダル RAG
- [ ] 7️⃣ フィードバック統合

**効果**:
- 長期的な知識ベース成長

---

## 🎯 最終的な AI Chat フロー（全改善実装後）

```
ユーザー: 「製造業の審査で Q-Risk が高い場合の対応方法は？」
  ↓
【Phase 1: 会話コンテキスト抽出】
- 過去のメッセージから「製造業」という文脈を認識
- 会話履歴から「Q-Risk について既に何か言及されたか」確認

  ↓
【Phase 2: 動的 RAG 検索】
- 「製造業 + Q-Risk 対応」で検索
- リアルタイムに最新の RAG 結果を取得
- 信頼度スコア付き（🟢 高信頼 / 🟡 中 / 🔴 低）

  ↓
【Phase 3: コンテキスト圧縮】
- RAG 結果を効率的に圧縮（1000 tokens → 200 tokens）
- 最重要部分のみを抽出

  ↓
【Phase 4: 動的プロンプト生成】
- RAG 結果が「Q-Risk 関連」と認識
- AI 役割を「リスク分析の専門家」に自動切り替え
- プロンプトを最適化

  ↓
【Phase 5: AI が回答生成】
「製造業における Q-Risk が高い場合の対応方法：

1. 即座の調査項目：
   - [[物件ファイナンス検索索引]]：残価評価再確認
   - [[業種別ベンチマーク]]：業界平均との比較

2. リスク軽減策：
   ...

📚 参考ノート：
🟢 高信頼 (0.92) | Projects/tune_lease_55/Asset Knowledge/物件ファイナンス検索索引.md
🟡 中信頼 (0.71) | AI Chat/2026-05-25（5日前）
」

  ↓
【Phase 6: ユーザーフィードバック】
「👍 役に立った | 👎 役に立たなかった」
  ↓ もし「役に立たなかった」なら
    - RAG スコアを自動調整
    - Gap 検出システムが「Q-Risk 対応情報の拡充」を改善候補化

  ↓
【翌朝】
- 朝の見直しで「Q-Risk 対応ガイド拡充」が改善候補に
- スコア付け → TOP 3 に上がり、実装が検討される
```

---

## 💡 期待される効果

| 指標 | 改善前 | 改善後 | 向上率 |
|------|-------|-------|--------|
| **回答の正確性** | 基本的 | RAG + 動的生成 | +40% |
| **ユーザー信頼度** | 不明確 | スコア表示で可視化 | +60% |
| **会話の自然さ** | 断片的 | コンテキスト継続 | +50% |
| **LLM コスト** | 高 | 圧縮で削減 | -30% |
| **Vault 成長** | 手動 | 自動 Gap 検出 | ∞ |

---

## 🚀 実装優先度まとめ

```
【すぐ実装】（優先度 🔴 高）
1️⃣ リアルタイム RAG 統合 → 即効性
8️⃣ Gap 検出システム → 継続改善の源

【次に実装】（優先度 🟡 中）
2️⃣ コンテキスト圧縮 → コスト最適化
3️⃣ 会話履歴統合 → UX 向上
4️⃣ 信頼度スコア → 透明性
6️⃣ 動的プロンプト → 回答品質

【後で実装】（優先度 🟢 低）
5️⃣ マルチモーダル RAG → 高度な機能
7️⃣ フィードバック統合 → 長期的改善
```

---

**次のステップ**: Phase 1 の実装を提案します。1️⃣ リアルタイム RAG 統合 と 8️⃣ Gap 検出システム を実装しますか？
