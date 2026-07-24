# 審査ドメインのシーム — 紫苑を業種横断で使うための境界

紫苑（人格エンジン）を別業種の審査システムへ載せ替えるための、ドメイン境界（seam）の
スケルトン。背景と全体像は「紫苑 移植可能性マップ」を参照。ここでは**コードに落ちた契約**を記す。

## 何のためか

紫苑のドメイン依存は、実質 `lease_intelligence_tools.py`（ツール）と
`lease_intelligence_dialogue.py`（知識注入）の 2 点に集中している。この 2 点を
「審査ドメインが提供する契約」に還元しておけば、業種ごとに Provider を差し替えるだけで
紫苑を住まわせられる。

## 契約（`screening_domain/`）

- `screening_domain/contract.py`
  - `Thresholds(approval, conditional, review)` — 判定ライン。`band(score)` で
    approved/conditional/review/rejected に変換。**承認ライン集約の受け皿**。
  - `DomainProvider`(Protocol) — 審査ドメインの最小契約：
    `thresholds / score / search_cases / inspect_policy / coefficients /
    lookup_rules / search_knowledge / benchmark / portfolio_stats`。
  - `build_tool_registry(provider)` — 紫苑のドメイン依存ツール名 → provider メソッドの
    写像。将来 `execute_tool` をここ経由に差し替える差し替え口の雛形。
  - `DOMAIN_TOOL_NAMES` — レジストリ対象（＝ドメインに触れる）ツール名。中立ツール
    （`get_pipeline_status` / `search_obsidian` / git 系）は人格側に残す。
- `screening_domain/lease_provider.py`
  - `LeaseDomainProvider` — リースを 1 インスタンスに束ねた参照実装（既存コードへの
    薄い遅延アダプタ。審査ロジックは複製・変更しない）。
  - `get_active_provider()` — 現状はリース固定。将来 `SCREENING_DOMAIN` 等で切替。

## 現状はスケルトン（ライブ配線はまだ）

- ライブの `lease_intelligence_tools.execute_tool` は**まだ差し替えていない**。
  次段で `build_tool_registry(get_active_provider())` に配線する。
- 対話の知識注入（`lease_finance_knowledge`）も次段で `provider.search_knowledge` 経由に。

## 承認ライン集約の現状

- 判定ラインの**定義は `constants.py` の 1 箇所**（`scoring_core` は re-export のみ）。
  `tests/test_threshold_single_source.py` が AST でこの不変条件をロックする。
- ドメイン契約の `Thresholds` も constants の値を通す（別勘定の数値を持たない）。

### 要 Plan-First の残課題（本PR対象外）

審査判定に効く**数値ハードコード**が一部モジュールに残っている（スコアリング隣接＝
CLAUDE.md の Plan-First 対象のため、本PRでは触れない）。単一ソース化するなら承認を得てから：

- `credit_limit.py`（`score < 60`）
- `secondary_review.py`（`score >= 60` / `score < 75`）
- `lease_intelligence_mind.py`（`score >= 60`）

※ 可視化・別指標（Q_risk、再製造スコア、チャート色分け）の 60/40 は判定ラインではないため対象外。

## 別業種へ載せ替える手順（要約）

1. 新業種用に `XxxDomainProvider` を書く（`DomainProvider` を実装）。
2. `get_active_provider()` を業種切替に対応させる。
3. `execute_tool` をレジストリ経由に配線（本スケルトンの次段）。
4. 知識注入を `provider.search_knowledge` 経由に。
5. 記憶アイデンティティの設計（1 人の紫苑が複数業種 / 業種ごとにクローン）。
