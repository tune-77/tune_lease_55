---
name: auto-improvement-pipeline
description: 自律型改善リファクタリング・パイプライン。チャットログから改善点を自動抽出→妥当性検証→自動修正・Obsidian同期する3段階パイプラインシステム。「パイプライン実行」「改善を自動化」「リファクタリング」などのキーワードで使用。
---

# auto-improvement-pipeline スキル

人間とのチャットログから「システム改善点」を自動的に検知・抽出し、構造化・妥当性検証を経て、コードやObsidianへノンストップで自動反映させるスキルです。

## 概要

### 処理フロー

```
【チャットログ】
    ↓
【Step 1】改善点抽出・構造化
  - [改善], [TODO] などのトリガーを検出
  - 改善内容を JSON に正規化
    ↓
【Step 2】妥当性検証
  - 論理的整合性チェック
  - エッジケース検査
  - テスト破壊リスク評価
    ↓
【Step 3】自動修正・デプロイ
  - コード自動適用
  - ローカルテスト実行
  - Obsidian ナレッジ同期
  - Git コミット・プッシュ
    ↓
【完了】
```

### 安全装置（ガードレール）

- **自動テスト駆動**: テスト不合格 → ロールバック
- **検証二段階**: Codex抽出 → 独立チェッカー検証
- **Obsidian トレーサビリティ**: 全変更をナレッジノート化

---

## 使用方法

### 前提条件

- Python 3.10+
- pytest（ローカルテスト実行用）
- Git（コミット・プッシュ用）
- Obsidian Vault（ナレッジ同期用）

### コマンドライン実行

#### 1. 基本的な実行（検証のみ）

```bash
python pipeline_runner.py chat_log.txt --dry-run
```

- チャットログ内の改善点を抽出・検証します
- 実際のコード修正は行いません
- JSON 形式の結果を表示

#### 2. 本実行（改善を自動適用）

```bash
python pipeline_runner.py chat_log.txt
```

- Step 1～3 を実行
- 検証済み改善を自動修正・デプロイ
- Git コミット・Obsidian 同期も実行

#### 3. Obsidian RAGコンテキストを指定

```bash
python pipeline_runner.py chat_log.txt \
  --rag-context "RAGから取得した既存仕様テキスト" \
  --output result.json
```

- RAG コンテキストを渡すことで、既存仕様との競合を検査
- 結果を JSON ファイルに保存

#### 4. ワークスペースを指定

```bash
python pipeline_runner.py chat_log.txt \
  --workspace /path/to/tune_lease_55
```

---

## チャットログのフォーマット

パイプラインが改善点を検出するため、以下のトリガーを使用してください：

### トリガーキーワード

```
[改善], [TODO], [FIX], [REFACTOR], [バグ], [問題]
```

### 例1: シンプルな改善指示

```
ユーザー: [改善] quantum_analysis_module.py の閾値を32に下げる

現在は quantum_risk >= 35 で要注意フラグが立ちます。
テストデータを見ると、32に下げるべきです。

理由：実際のリスク案件が MEDIUM で誤判定されています。
```

### 例2: 複数の改善指示

```
ユーザー: 審査ロジックに問題があります。

[改善] grade_normalizer.py で「無格付」を正しくハンドル
現在は例外を落としていますが、デフォルト値を適用すべき。

[TODO] asset_scorer.py のマジックナンバーを定数化
理由：保守性向上＆テスト時に値を動的に変更できない
```

---

## 出力フォーマット

### Step 1 出力（改善案の JSON）

```json
[
  {
    "id": "REV-001",
    "target_module": "quantum_analysis_module.py",
    "title": "quantum_risk の閾値を35から32に下げる",
    "description": "現在は quantum_risk >= 35 で要注意フラグが立ちます...",
    "reason": "実際のリスク案件が MEDIUM で誤判定されています",
    "priority": "HIGH"
  }
]
```

### Step 2 出力（検証結果）

```json
{
  "status": "APPROVED",
  "verification_report": "改善案ID: REV-001...",
  "critical_flaws": [],
  "alternative_suggestion": null
}
```

### Step 3 出力（適用結果）

```json
{
  "status": "COMPLETED",
  "applied_count": 1,
  "failed_count": 0,
  "commit_result": {
    "success": true,
    "commit_hash": "abc1234...",
    "message": "コミット成功"
  },
  "applied_improvements": [...]
}
```

---

## 検証ルール（Step 2）

### 承認（APPROVED）条件

- 致命的フローなし
- ターゲットファイルが存在
- エッジケースを適切に処理
- テスト破壊の可能性がない

### 拒否（REJECTED）条件

- ゼロ除算やNULL参照のリスク
- マジックナンバー（根拠なき数値）
- テスト破壊可能性
- スコープ不明確

---

## トラブルシューティング

### 改善点が抽出されない

✅ チャットログに `[改善]`, `[TODO]` などのトリガーキーワードを含めてください。

```diff
- 「このロジックを修正してください」
+ 「[改善] このロジックの～を修正する」
```

### 検証が失敗する

✅ RAGコンテキストを確認してください。既存仕様との衝突がないか確認します。

```bash
python pipeline_runner.py chat_log.txt \
  --rag-context "既存仕様のテキスト" \
  --dry-run
```

### テスト失敗でロールバック

✅ コード修正パッチの品質を確認してください。LLMが生成したパッチに手動修正が必要な場合があります。

---

## セットアップ

### フォルダ構成

```
.agents/skills/auto-improvement-pipeline/
  ├── SKILL.md（このファイル）
  ├── pipeline_runner.py（メインエンジン）
  └── scripts/
      ├── step1_extract_and_structure.py
      ├── step2_validation_checker.py
      └── step3_auto_apply.py
```

### 依存モジュール

- `obsidian_ai_context.py` — Obsidian RAG コンテキスト取得
- `obsidian_query.py` — Obsidian 検索キーワード正規化
- `mobile_app/obsidian_bridge.py` (または `obsidian_bridge.py`) — Obsidian Vault 統合

---

## 注意事項

⚠️ **自動修正について**

Step 3 の「コード自動修正」は現在、プレースホルダー実装です。本番運用では、以下の実装が必要です：

1. **LLM (Gemini API) 統合**: `step3_auto_apply.py` の `_generate_code_patch()` 関数に LLM コード生成ロジックを組み込む
2. **Diff 適用**: 生成されたパッチを実際のファイルに apply

```python
# 例：Gemini API を使用したコード修正生成
import anthropic

def _generate_code_patch_with_llm(target_file, improvement):
    client = anthropic.Anthropic()
    
    with open(target_file) as f:
        current_code = f.read()
    
    prompt = f"""
    以下のファイルに対して、示された改善を適用するPythonコードのdiffを生成してください。
    
    ファイル: {target_file}
    改善内容: {improvement['description']}
    
    出力形式: unified diff
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text
```

---

## 今後の拡張

- [ ] Web UI（Streamlit）での改善ログ閲覧・承認
- [ ] スラッシュコマンド統合（Slack `/ auto-refactor`）
- [ ] 改善品質スコアリング（人間の改善承認率に基づく）
- [ ] マルチモーダル改善（テキスト＋画像の修正指示）

---

## サポート

問題が発生した場合は、以下を確認してください：

- チャットログのフォーマット確認
- RAGコンテキストの妥当性
- ローカル pytest の実行状況
- Obsidian Vault の存在確認
- Git の状態（コミット権限、リモート設定）
