"""
改善ルールアプライヤー。

完全実装: patch_json, scoring_weight, ui_text
スタブ:    add_api_field, endpoint_add, config_value, llm_diff
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .schema import ApplyResult, ImprovementRule

# プロジェクトルート = このファイルの 3 階層上
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# scoring_weight の保存先
_SCORING_WEIGHTS_PATH = _PROJECT_ROOT / "api" / "scoring_weights.json"
# ui_text の保存先
_UI_LABELS_PATH = _PROJECT_ROOT / "frontend" / "src" / "lib" / "ui_labels.json"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_rule(rule: dict) -> ApplyResult:
    """台帳の 1 ルールを適用して ApplyResult を返す。"""
    r = ImprovementRule.from_dict(rule)
    dispatch = {
        "patch_json":      _apply_patch_json,
        "add_api_field":   _stub("add_api_field"),
        "ui_text":         _apply_ui_text,
        "scoring_weight":  _apply_scoring_weight,
        "endpoint_add":    _stub("endpoint_add"),
        "config_value":    _stub("config_value"),
        "llm_diff":        _stub("llm_diff"),
    }
    handler = dispatch.get(r.type)
    if handler is None:
        raise ValueError(f"Unknown rule type: {r.type!r}")
    return handler(r)


# ---------------------------------------------------------------------------
# patch_json — 完全実装
# ---------------------------------------------------------------------------

def _apply_patch_json(rule: ImprovementRule) -> ApplyResult:
    """
    JSON ファイル内の要素を探して patch する。

    対応構造:
      - ルートが list の場合: 各要素（dict）を match 条件で絞り込み patch を適用
      - ルートが dict で "categories" キーを持つ場合: categories[*].items[*] を探索
      - ルートが dict の場合: ルート dict に直接 patch を適用
    """
    if not rule.target:
        return ApplyResult(rule.rev_id, False, "target が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    target_path = _PROJECT_ROOT / rule.target
    if not target_path.exists():
        return ApplyResult(rule.rev_id, False, f"ファイルが存在しません: {target_path}")

    with open(target_path, encoding="utf-8") as f:
        data = json.load(f)

    original = copy.deepcopy(data)
    matched = 0

    if isinstance(data, list):
        matched = _patch_list(data, rule.match, rule.patch)
    elif isinstance(data, dict):
        if "categories" in data:
            matched = _patch_categories(data["categories"], rule.match, rule.patch)
        else:
            if _dict_matches(data, rule.match):
                for k, v in rule.patch.items():
                    data[k] = v
                matched = 1

    if matched == 0:
        return ApplyResult(
            rule.rev_id, False,
            f"match 条件に合う要素が見つかりませんでした: {rule.match}"
        )

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    diff = _build_diff_summary(original, data, rule.target)
    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"{matched} 件の要素を更新しました",
        changed_file=rule.target,
        diff_summary=diff,
    )


def _patch_list(lst: list, match: dict | None, patch: dict) -> int:
    count = 0
    for item in lst:
        if isinstance(item, dict) and _dict_matches(item, match):
            for k, v in patch.items():
                item[k] = v
            count += 1
    return count


def _patch_categories(categories: list, match: dict | None, patch: dict) -> int:
    """categories[*].items[*] 構造（useful_life_equipment.json 等）を再帰探索。"""
    count = 0
    for cat in categories:
        items = cat.get("items", [])
        count += _patch_list(items, match, patch)
        # ネストした subcategories があれば再帰
        if "subcategories" in cat:
            count += _patch_categories(cat["subcategories"], match, patch)
    return count


def _dict_matches(d: dict, match: dict | None) -> bool:
    if not match:
        return True
    return all(d.get(k) == v for k, v in match.items())


def _build_diff_summary(original: Any, updated: Any, label: str) -> str:
    """変更箇所を人間が読める形でまとめる（差分が大きすぎる場合は省略）。"""
    lines = [f"--- {label} (before)", f"+++ {label} (after)"]
    _collect_diffs(original, updated, lines, path="")
    return "\n".join(lines) if len(lines) > 2 else "(差分なし)"


def _collect_diffs(a: Any, b: Any, out: list, path: str) -> None:
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(list(a.keys()) + list(b.keys())):
            _collect_diffs(a.get(k), b.get(k), out, f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list):
        for i, (ai, bi) in enumerate(zip(a, b)):
            _collect_diffs(ai, bi, out, f"{path}[{i}]")
    else:
        out.append(f"  {path}: {a!r} → {b!r}")


# ---------------------------------------------------------------------------
# scoring_weight — JSON ファイルへのスコアリング重み upsert
# ---------------------------------------------------------------------------

_VALID_SCORING_TARGETS = {"ASSET_WEIGHT", "CATEGORY_SCORE_ITEMS"}


def _apply_scoring_weight(rule: ImprovementRule) -> ApplyResult:
    """
    api/scoring_weights.json にスコアリング重みを upsert し、
    category_config.py の _load_scoring_overrides() 経由でスコアラーに反映する。

    scoring_weights.json エントリ形式（category_config.py が読む形式）:
      ASSET_WEIGHT 上書き:
        match: {"target": "ASSET_WEIGHT", "category": "車両", "param": "asset_w"}
        patch: {"value": 0.40}

      CATEGORY_SCORE_ITEMS 上書き:
        match: {"target": "CATEGORY_SCORE_ITEMS", "category": "IT機器",
                "item_id": "tech_obsolescence", "param": "weight"}
        patch: {"value": 25}

    match に target が必須。patch に value が必須。
    """
    if not rule.match:
        return ApplyResult(rule.rev_id, False, "match が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    target = rule.match.get("target")
    if target not in _VALID_SCORING_TARGETS:
        return ApplyResult(
            rule.rev_id, False,
            f"match.target が未サポートです: {target!r}。"
            f"有効値: {sorted(_VALID_SCORING_TARGETS)}"
        )
    if "value" not in rule.patch:
        return ApplyResult(rule.rev_id, False, "patch に 'value' キーが必要です")

    if _SCORING_WEIGHTS_PATH.exists():
        with open(_SCORING_WEIGHTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    original = copy.deepcopy(data)

    # match の全キーが一致するエントリを探す
    matched_idx: int | None = None
    for i, entry in enumerate(data):
        if isinstance(entry, dict) and _dict_matches(entry, rule.match):
            matched_idx = i
            break

    if matched_idx is not None:
        data[matched_idx]["value"] = rule.patch["value"]
        action = "更新"
    else:
        new_entry: dict = dict(rule.match)
        new_entry["value"] = rule.patch["value"]
        data.append(new_entry)
        action = "新規追加"

    with open(_SCORING_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    diff = _build_diff_summary(original, data, "api/scoring_weights.json")
    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"scoring_weights.json に 1 件を{action}しました（{target}: {rule.match}）",
        changed_file="api/scoring_weights.json",
        diff_summary=diff,
    )


# ---------------------------------------------------------------------------
# ui_text — JSON ファイルへの UI ラベル upsert
# ---------------------------------------------------------------------------

def _apply_ui_text(rule: ImprovementRule) -> ApplyResult:
    """
    frontend/src/lib/ui_labels.json に UI テキスト・ラベルを upsert する。

    - match 条件に合うエントリがあれば patch で上書き
    - なければ match + patch を合わせた新規エントリを追加
    - ファイルが存在しない場合は空リストで新規作成
    """
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    if _UI_LABELS_PATH.exists():
        with open(_UI_LABELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        _UI_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = []

    original = copy.deepcopy(data)

    matched_idx: int | None = None
    if rule.match:
        for i, entry in enumerate(data):
            if isinstance(entry, dict) and _dict_matches(entry, rule.match):
                matched_idx = i
                break

    if matched_idx is not None:
        for k, v in rule.patch.items():
            data[matched_idx][k] = v
        action = "更新"
    else:
        new_entry = dict(rule.match or {})
        new_entry.update(rule.patch)
        data.append(new_entry)
        action = "新規追加"

    with open(_UI_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    diff = _build_diff_summary(original, data, "frontend/src/lib/ui_labels.json")
    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"ui_labels.json に 1 件を{action}しました",
        changed_file="frontend/src/lib/ui_labels.json",
        diff_summary=diff,
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _stub(type_name: str):
    def _handler(rule: ImprovementRule) -> ApplyResult:
        raise NotImplementedError(
            f"rule type '{type_name}' はまだ実装されていません（REV-103 PoC スコープ外）"
        )
    return _handler
