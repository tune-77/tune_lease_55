# Obsidian RAG 改善ガイド

本ドキュメントは、`obsidian_bridge.py` と `obsidian_bridge_enhancements.py` による7つの改善を説明します。

---

## 🎯 7つの改善概要

### 1️⃣ リトライロジック & エラーハンドリング ✅

**ファイル**: `obsidian_bridge_enhancements.py`（`get_vector_store_with_retry`）

Vector Store API の呼び出しに tenacity による自動リトライを適用。

```python
from obsidian_bridge_enhancements import get_vector_store_with_retry

# 最大3回、指数バックオフで自動リトライ
store = get_vector_store_with_retry()
```

**効果**: API 一時的な失敗に強くなった。

---

### 2️⃣ Frontmatter メタデータ抽出 ✅

**ファイル**: `obsidian_bridge_enhancements.py`（`extract_frontmatter`, `extract_metadata`）

YAML frontmatter から以下を自動抽出：
- `title`（ノートタイトル）
- `tags`（タグ）
- `industry`（業種）
- `score_range`（スコア範囲）
- `credit_rating`（信用格付）

**使用例**:
```yaml
---
title: 製造業 A社 - スコア75
tags: [製造, 大型機械, 高リスク]
industry: c 製造業
score_range: [70, 80]
credit_rating: 4-6
---
```

```python
from obsidian_bridge_enhancements import extract_metadata
from pathlib import Path

path = Path("Notes/製造業A社.md")
text = path.read_text()
metadata = extract_metadata(path, text)
# {
#     "title": "製造業 A社 - スコア75",
#     "tags": ["製造", "大型機械", "高リスク"],
#     "industry": "c 製造業",
#     "score_range": (70, 80),
#     ...
# }
```

**効果**: 検索時にメタデータでフィルタ・ボーナススコアを適用可能。

---

### 3️⃣ BM25 ランキング ✅

**ファイル**: `obsidian_bridge_enhancements.py`（`BM25Scorer`）

BM25 アルゴリズムで検索結果の関連性をスコアリング。

```python
from obsidian_bridge_enhancements import BM25Scorer

scorer = BM25Scorer()
documents = [doc1, doc2, doc3, ...]
scorer.fit(documents)  # IDF学習

score = scorer.score("スコア 72 製造業", document)
```

**効果**: 単純なマッチより関連性の高い結果が上位に。

---

### 4️⃣ ファイルハッシュベース差分更新 ✅

**ファイル**: `obsidian_bridge_enhancements.py`（`_file_hash`, `_needs_update`, `prune_stale_cache`）

Vault インデックス再構築時、SHA256 ハッシュで変更ファイルのみ更新。

```python
from obsidian_bridge_enhancements import _needs_update, prune_stale_cache
from pathlib import Path

vault = Path("~/Obsidian Vault")

# 削除されたファイルをキャッシュから削除
prune_stale_cache(vault)

# 個別ファイルの更新判定
if _needs_update(vault / "Notes/case1.md"):
    # 再インデックス
    ...
```

**効果**: インデックス更新が**数倍高速化**（フルスキャン不要）。

---

### 5️⃣ 業種・スコア範囲フィルタ ✅

**ファイル**: `obsidian_bridge.py` & `obsidian_bridge_enhancements.py`

#### 5a. 業種フィルタ
```python
from obsidian_bridge import search_notes_with_industry_filter

# 製造業（'c'）に限定した検索
results = search_notes_with_industry_filter(
    query="スコア 72",
    industry_code="c",  # 製造業
    limit=4
)
```

#### 5b. スコア範囲フィルタ
```python
from obsidian_bridge import search_cases_by_score_range

# 現在案件（スコア70-80）の過去事例を検索
results = search_cases_by_score_range(
    query="製造業",
    min_score=70.0,
    max_score=80.0,
    limit=4
)
```

**効果**: 関連性の高い過去案件が自動的にピックアップされる。

---

### 6️⃣ Wikilink トラバーサル（リンク先も検索対象） ✅

**ファイル**: `obsidian_bridge_enhancements.py`（`extract_wikilinks`, `prefetch_wikilinks`）

```python
from obsidian_bridge import search_with_wikilink_context

results = search_with_wikilink_context(
    query="車両リース 残価",
    limit=4
)

# 結果:
# {
#     "path": "Asset Knowledge/車両ファイナンス.md",
#     "snippet": "...",
#     "linked_context": {
#         "[[残価設定法]]": "残価とは...",
#         "[[中古相場データ]]": "2026年の...",
#     }
# }
```

**効果**: リンク先ノートの内容もコンテキストに含まれるため、より深い知識検索が可能。

---

### 7️⃣ リアルタイム Vault 同期（オプション）

**推奨ツール**: `watchdog` ライブラリ（未導入）

```bash
# 必要に応じてインストール
pip install watchdog
```

**使用例**（将来実装）:
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class VaultWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        # .md ファイル変更を検知 → インデックス更新
        if event.src_path.endswith('.md'):
            update_index_for_file(event.src_path)

observer = Observer()
observer.schedule(VaultWatcher(), vault_path, recursive=True)
observer.start()
```

**効果**: ユーザーが Vault を編集すると、リアルタイムでシステムに反映。

---

## 📚 API リファレンス

### obsidian_bridge.py（改善版）

#### `search_notes(query, limit=4, max_chars=700)`
基本検索（既存 + リトライ）

#### `search_notes_with_industry_filter(query, industry_code, limit=4)` ✨
業種フィルタ付き

#### `search_cases_by_score_range(query, min_score, max_score, limit=4)` ✨
スコア範囲フィルタ付き

#### `search_with_wikilink_context(query, limit=4)` ✨
Wikilink トラバーサル付き

---

### obsidian_bridge_enhancements.py

#### Frontmatter & メタデータ
- `extract_frontmatter(text)` → `dict`
- `extract_metadata(path, text)` → `dict`

#### BM25 ランキング
- `BM25Scorer` クラス
  - `.fit(documents)` — IDF 学習
  - `.score(query, document)` → `float`

#### 差分更新
- `_file_hash(path)` → `str`
- `_needs_update(path)` → `bool`
- `_update_hash_cache(path)` → `None`
- `prune_stale_cache(vault)` → `None`

#### フィルタリング
- `filter_by_industry(notes, industry_code)` → `list`
- `filter_by_score_range(notes, min_score, max_score)` → `list`

#### Wikilink
- `extract_wikilinks(text, vault)` → `list[Path]`
- `prefetch_wikilinks(path, vault, max_depth=1)` → `dict`

#### リトライ
- `get_vector_store_with_retry()` — リトライ付き Vector Store 取得

---

## 🚀 統合例：AI Chat での使用

```python
# api/knowledge/chat_handler.py など

from mobile_app.obsidian_bridge import (
    search_with_wikilink_context,
    search_cases_by_score_range,
    search_notes_with_industry_filter,
)

async def handle_case_analysis(case_data: dict):
    """案件分析時に Obsidian から関連情報を検索。"""
    industry_code = case_data.get("industry_code")  # 'c', 'd', etc.
    current_score = case_data.get("current_score")   # 72.5
    query = case_data.get("analysis_query")         # "スコア算出根拠"

    # 1. 関連する過去案件
    similar_cases = search_cases_by_score_range(
        query=industry_code,
        min_score=current_score - 10,
        max_score=current_score + 10,
        limit=3
    )

    # 2. 業種別ガイドライン
    guidelines = search_notes_with_industry_filter(
        query=query,
        industry_code=industry_code,
        limit=2
    )

    # 3. 関連知識（リンク先も含む）
    knowledge = search_with_wikilink_context(
        query=query,
        limit=2
    )

    # コンテキスト構築
    context = {
        "similar_cases": similar_cases,
        "guidelines": guidelines,
        "knowledge": knowledge,
    }

    return context
```

---

## 💡 ベストプラクティス

### 1. Frontmatter の活用
すべてのノート冒頭に frontmatter を追加：

```yaml
---
title: 製造業スコアリングガイド
tags: [製造, スコアリング, Q-Risk]
industry: c 製造業
created: 2026-05-30
updated: 2026-05-30
---
```

### 2. Wikilink の積極的活用
関連ノートへのリンクを張ることで、自動的に知識チェーンが構築される：

```markdown
## 参照リンク
- [[残価設定法]]
- [[中古相場データ]]
- [[業種別ベンチマーク]]
```

### 3. Cases/ フォルダの構造
```
Cases/
├── 2026-05/
│   ├── 製造_A社_スコア75.md
│   ├── 建設_B社_スコア68.md
│   └── ...
├── 2026-04/
└── ...
```

frontmatter に `score_range` を記載すると、スコア範囲検索で自動マッチ。

### 4. タグの活用
```
#製造 #高スコア #条件付き #q-risk-高
```

タグはメタデータに自動抽出され、検索ランキングのボーナスに使用。

---

## ⚠️ 既知の制限と今後の改善

| 項目 | 現状 | 改善予定 |
|------|------|--------|
| Wikilink トラバーサル | 1段階のみ | 無制限チェーン |
| リアルタイム同期 | TTL 5分 | inotify/watchdog |
| フルテキスト検索 | キーワード | セマンティック埋め込み |
| インデックスサイズ | 無制限 | パフォーマンス最適化 |

---

## 🧪 テスト方法

```bash
# 拡張モジュールのテスト
python -m pytest mobile_app/test_obsidian_enhancements.py -v

# 統合テスト
python -c "
from mobile_app.obsidian_bridge import search_notes_with_industry_filter
results = search_notes_with_industry_filter('製造', 'c', limit=2)
print(results)
"
```

---

## 📝 トラブルシューティング

### Vector Store 連続失敗
- リトライが3回失敗した場合、キーワード検索にフォールバック
- ログ: `logging.debug(f'Vector store search failed: {e}, falling back...')`

### Frontmatter 解析失敗
- PyYAML がない場合は `{}` を返す
- `yaml` ライブラリが必須

### パフォーマンス低下
- `prune_stale_cache()` で削除ファイルのキャッシュを削除
- 大型 Vault（>10000 ファイル）では差分更新がより顕著な効果

---

**作成日**: 2026-05-30
**バージョン**: 1.0
