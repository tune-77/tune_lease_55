---
spec_id: P7-001
phase: 7
title: リース業界ドメイン辞書（lease_domain_glossary）
status: draft
author: Claude Fable
reviewer: ""
version: "1.0"
created: 2026-07-09
updated: 2026-07-09
depends_on: []
superseded_by: ""
---

# P7-001 — リース業界ドメイン辞書（lease_domain_glossary）

---

## 1. Goal

リース審査ドメイン固有の同義語・専門用語を1つのJSON辞書に集約し、RAG検索（P7-002 クエリ拡張）が「飲食業の赤字企業」と「飲食店の営業赤字」を同じ意図として扱えるようにする。辞書は検索精度改善の基盤データであり、それ自体はスコアリング・審査判定に一切影響しない。

> 出典: `代替案実装計画_精度改善Ver2.md` 代替案A「ドメイン辞書改善」の詳細設計。

---

## 2. Scope

### In scope
- 辞書データファイル `static_data/lease_domain_glossary.json` のスキーマ定義と初期シード（シノニム8グループ以上・業界用語5語以上）
- 読み込みモジュール `api/knowledge/domain_glossary.py`（ロード・検証・シノニム引き）
- 単体テスト `tests/test_domain_glossary.py`

### Out of scope
- 検索クエリ拡張ロジックと `KnowledgeVectorStore` への統合（P7-002）
- 埋め込みモデルの変更（代替案B。本SPECとは独立）
- `scoring_core.py` / 審査スコアへの影響（辞書は検索専用）
- 辞書の自動学習・自動更新（将来のフィードバック統合で検討）

### 計画書からの変更点（レビュー時に確認してください）
1. **配置**: 計画書は `mobile_app/lease_domain_glossary.json` を指定しているが、現行の本番RAGは `api/knowledge/vector_store.py`（Next.js + FastAPI 経路）であるため、両スタックから参照できる `static_data/` に置く。`static_data/` は業種・物件マスタ（`lease_assets.json`・`industry_hints.json` 等）の既存置き場であり整合する。
2. **キー名**: 計画書のJSON例は日本語キー（`シノニムグループ`）だが、コードからの参照安定性のため英語キー（`synonym_groups`）とする。

---

## 3. Inputs / Outputs

### Inputs（辞書ファイル）
| 項目名 | 型 | 必須/任意 | 説明 |
|-------|-----|----------|------|
| version | int | 必須 | スキーマバージョン（初版は 1） |
| updated | str | 必須 | 最終更新日 `YYYY-MM-DD` |
| synonym_groups | list[SynonymGroup] | 必須 | シノニムグループの配列 |
| industry_terms | list[IndustryTerm] | 必須 | 業界用語（略語・専門語の説明）の配列 |

### Outputs（ローダーAPI）
| 項目名 | 型 | 説明 |
|-------|-----|------|
| get_glossary() | Glossary | 検証済み辞書（モジュールレベルキャッシュ、mtime変更時のみ再読込） |
| synonyms_for(term) | list[tuple[str, float]] | term が canonical または synonym に一致するグループの (別語, weight) 一覧 |
| known_terms() | frozenset[str] | canonical + synonym の全語彙（クエリ走査の高速判定用） |

---

## 4. Data Model

```python
class SynonymGroup(TypedDict):
    canonical: str        # 代表語（例: "赤字企業"）
    synonyms: list[str]   # 同義語（例: ["営業赤字", "当期赤字", "欠損"]）
    weight: float         # 拡張時の信頼度 0.0-1.0（P7-002 でスコア減衰に使用）
    source: str           # 根拠（出典ファイル名 or "業務知識"）

class IndustryTerm(TypedDict):
    term: str             # 用語（例: "Q-Risk"）
    meaning: str          # 意味の短文説明
    context: str          # 使われる文脈（例: "審査基準"）
    aliases: list[str]    # 表記ゆれ（例: ["Qリスク", "q_risk"]）
```

### 初期シードの構成と出典（Cite the Source）

| 区分 | シード内容 | 出典 |
|------|----------|------|
| 財務用語 | 赤字企業 / 自己資本比率 / 残価 / デフォルト（延滞・債務不履行） | `代替案実装計画_精度改善Ver2.md` の例示 + `static_data/industry_averages.json`（財務指標名） |
| 物件用語 | リース物件 / 建設機械（建機・重機・ユンボ）/ IT・OA機器 / 車両 | `static_data/lease_assets.json` の items（建設機械・IT・OA機器 等の取扱物件名） |
| 機種分類 | 電子計算機（サーバ・パソコン）等の機種名と例示語 | `期待使用期間.json` の usage_period_data（category / item_name / examples） |
| 業種用語 | 飲食業 / 総合工事業・職別工事業（大工・とび等） | `static_data/industry_hints.json` のキー（JSIC業種名） |
| 契約・会計用語 | ファイナンスリース / オペレーティングリース / 再リース / 所有権移転 | `static_data/lease_classification.json` の classification_flow |
| 審査用語 | 審査基準 / 与信基準、Q-Risk / LTV / EBITDA | `代替案実装計画_精度改善Ver2.md` の例示 + `.claude/rules/workflow.md`（Q_risk 閾値定義） |
| 耐用年数用語 | 法定耐用年数 / 耐用年数 / 償却年数 | `static_data/useful_life_equipment.json`（国税庁耐用年数表参照） |

※ 上記出典にない語をシードに加える場合は「これは推測です」と明示した上でレビューに委ねる（Freshman Rules / Cite the Source）。

---

## 5. API / Interface

```python
# api/knowledge/domain_glossary.py（新規）

GLOSSARY_PATH = os.path.join(_REPO_ROOT, "static_data", "lease_domain_glossary.json")

def get_glossary() -> dict:
    """検証済み辞書を返す。ファイル不在・破損時は空辞書構造を返し、検索は無拡張で継続する。"""

def synonyms_for(term: str) -> list[tuple[str, float]]:
    """term を含むシノニムグループの他の語を (語, weight) で返す。大文字小文字・NFKC正規化して照合。"""

def known_terms() -> frozenset[str]:
    """canonical + synonyms + industry_terms.term/aliases の全語彙。"""
```

---

## 6. Business Rules

**BR-701**: 辞書は検索専用
- 条件：常時
- 処理：辞書・ローダーは `api/knowledge/` 配下と検索経路からのみ参照する。スコアリング（`scoring_core.py` 等）から import しない
- 根拠：検索精度改善が目的であり、審査結果への影響を遮断する（`CLAUDE.md` スコープ厳守）

**BR-702**: 出典のない語を勝手に増やさない
- 条件：辞書へ語を追加するとき
- 処理：`source` フィールドに出典（`static_data/` のファイル名等）を必ず記録する。出典がない語は `source: "業務知識(要確認)"` とし、レビューを必須とする
- 根拠：Freshman Rules「Cite the Source」

**BR-703**: 読み込み失敗はフェイルセーフ
- 条件：辞書ファイルが不在・JSON破損・スキーマ不正のとき
- 処理：警告ログを出し、空の辞書（拡張なし）で継続する。例外を上位に投げない
- 根拠：RAG検索は辞書がなくても従来どおり動作しなければならない（後方互換）

---

## 7. UI / UX

フロントエンド変更なし（本SPECはデータ+ローダーのみ）。

---

## 8. Error Handling

| エラー条件 | 処理 | ユーザー向けメッセージ |
|-----------|------|---------------------|
| 辞書ファイル不在 | 空辞書で継続、`logger.warning` | （表示なし） |
| JSON パースエラー | 空辞書で継続、`logger.warning` | （表示なし） |
| weight が 0-1 範囲外 | 当該グループを 1.0 or 0.0 にクランプして読込 | （表示なし、ログに記録） |
| synonyms が空のグループ | 当該グループをスキップ | （表示なし、ログに記録） |

---

## 9. Acceptance Criteria

**AC-701**: 正常ロード
- Given: シード済み `static_data/lease_domain_glossary.json`
- When: `get_glossary()` を呼ぶ
- Then: synonym_groups が8件以上、industry_terms が5件以上返り、全グループに canonical / synonyms / weight / source がある

**AC-702**: シノニム引き（双方向）
- Given: 「赤字企業」グループ（synonyms に「営業赤字」を含む）
- When: `synonyms_for("赤字企業")` と `synonyms_for("営業赤字")` を呼ぶ
- Then: 前者は「営業赤字」を含み、後者は「赤字企業」を含む（canonical からも synonym からも引ける）

**AC-703**: ファイル不在フォールバック
- Given: 辞書ファイルが存在しないパスを指すローダー
- When: `get_glossary()` / `synonyms_for("赤字企業")` を呼ぶ
- Then: 例外は発生せず、空構造 / 空リストが返る

**AC-704**: 表記正規化
- Given: 辞書に「LTV」がある
- When: `synonyms_for("ltv")`（小文字・全角混在）を呼ぶ
- Then: NFKC 正規化により同一グループとして解決される

**AC-705**: キャッシュと再読込
- Given: 一度ロード済みの辞書
- When: ファイルを更新（mtime 変更）して再度 `get_glossary()` を呼ぶ
- Then: 新しい内容が返る（`vector_store.py` の `_maybe_reload_ranking_config` と同じ mtime 方式）

---

## 10. Non-Functional Requirements

- **パフォーマンス**: `synonyms_for()` は 1ms 未満（辞書はロード時に逆引きインデックス化する）
- **後方互換性**: 既存の検索動作・APIレスポンスを一切変更しない（本SPEC単体では消費者がいない）
- **サイズ上限**: 辞書は当面 200 グループ以下を想定。超える場合は分割を再設計する
- **テストカバレッジ**: AC-701〜705 全件

---

## 11. Implementation Notes（実装者向け）

- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `api/main.py`（統合は P7-002 で行う）
- **参考にする既存実装**: `api/knowledge/vector_store.py` の `_maybe_reload_ranking_config()`（mtime ベース再読込）と `config/rag_ranking.json` の運用パターン
- **正規化**: `obsidian_query.split_query_terms` と同じく `unicodedata.normalize("NFKC", ...)` + lower() で照合する
- **新規ファイル**: `static_data/lease_domain_glossary.json`, `api/knowledge/domain_glossary.py`, `tests/test_domain_glossary.py`
- **mobile_app 側**: `mobile_app/hybrid_search.py` からも将来同じ辞書を参照できるが、本SPECでは mobile_app を変更しない

---

## 12. Test Plan

### 単体テスト
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_701 | AC-701 | シード辞書のロードとスキーマ検証 |
| test_702 | AC-702 | canonical / synonym 双方向引き |
| test_703 | AC-703 | ファイル不在・破損JSONのフェイルセーフ |
| test_704 | AC-704 | NFKC・大文字小文字の正規化照合 |
| test_705 | AC-705 | mtime 変更での再読込 |

### 回帰テスト
- 既存 `tests/test_rag_confidence.py` ほか RAG 関連テストが変化なくパスすること（本SPECは消費者を持たないため影響ゼロが期待値）

### 手動確認（実装後）
- [ ] `python3 -c "from api.knowledge.domain_glossary import get_glossary; print(len(get_glossary()['synonym_groups']))"` でシード件数が返る
