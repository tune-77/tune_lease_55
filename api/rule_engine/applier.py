"""
改善ルールアプライヤー。

完全実装: patch_json, scoring_weight, ui_text, add_api_field, endpoint_add,
          config_value, llm_diff
"""

from __future__ import annotations

import copy
import json
import os
import re
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
        "add_api_field":   _apply_add_api_field,
        "ui_text":         _apply_ui_text,
        "scoring_weight":  _apply_scoring_weight,
        "endpoint_add":    _apply_endpoint_add,
        "config_value":    _apply_config_value,
        "llm_diff":        _apply_llm_diff,
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

    # ASSET_WEIGHT ペア整合性チェック:
    # 同カテゴリの反対側パラメータが既にファイルに明示されている場合、
    # 合計が 1.0 になるか検証する（自動補正は _load_scoring_overrides 側が行う）
    new_value = rule.patch["value"]
    param = rule.match.get("param", "")
    if target == "ASSET_WEIGHT" and param in ("asset_w", "obligor_w"):
        other_param = "obligor_w" if param == "asset_w" else "asset_w"
        cat = rule.match.get("category")
        other_match = {"target": target, "category": cat, "param": other_param}
        other_entry = next(
            (e for e in data if isinstance(e, dict) and _dict_matches(e, other_match)),
            None,
        )
        if other_entry is not None:
            other_value = other_entry.get("value", 0)
            pair_sum = round(new_value + other_value, 10)
            if abs(pair_sum - 1.0) > 1e-9:
                return ApplyResult(
                    rule.rev_id,
                    False,
                    f"ASSET_WEIGHT ペア整合エラー: {cat} の {param}={new_value} + "
                    f"{other_param}={other_value} = {pair_sum} ≠ 1.0。"
                    "一方だけを指定すれば補数が自動計算されます。",
                )

    original = copy.deepcopy(data)

    # match の全キーが一致するエントリを探す
    matched_idx: int | None = None
    for i, entry in enumerate(data):
        if isinstance(entry, dict) and _dict_matches(entry, rule.match):
            matched_idx = i
            break

    if matched_idx is not None:
        data[matched_idx]["value"] = new_value
        action = "更新"
    else:
        new_entry: dict = dict(rule.match)
        new_entry["value"] = new_value
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
# add_api_field — Pydantic モデルへのフィールド追加
# ---------------------------------------------------------------------------

_VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _apply_add_api_field(rule: ImprovementRule) -> ApplyResult:
    """
    指定 Python ファイル内の Pydantic モデルクラスに新フィールドを追加する。

    patch キー:
      field_name  (str)  追加するフィールド名（必須）
      field_type  (str)  型アノテーション（省略時 "str"）
      default     (str)  デフォルト値のコード片（省略時 "None"）
      description (str)  Field(description=...) に渡す文字列（省略時は Field 不使用）
    """
    if not rule.target:
        return ApplyResult(rule.rev_id, False, "target が未指定です")
    if not rule.match:
        return ApplyResult(rule.rev_id, False, "match が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    model_name = rule.match.get("model", "")
    if not model_name:
        return ApplyResult(rule.rev_id, False, "match.model が未指定です")

    field_name = rule.patch.get("field_name", "")
    field_type = rule.patch.get("field_type", "str")
    default = rule.patch.get("default", "None")
    description = rule.patch.get("description", "")

    if not field_name:
        return ApplyResult(rule.rev_id, False, "patch.field_name が未指定です")
    if not field_type or not field_type.strip():
        return ApplyResult(rule.rev_id, False, "patch.field_type が空です")

    target_path = _PROJECT_ROOT / rule.target
    if not target_path.exists():
        return ApplyResult(rule.rev_id, False, f"ファイルが存在しません: {target_path}")

    source = target_path.read_text(encoding="utf-8")

    # クラス定義の存在確認
    class_def_re = re.compile(
        r"^class\s+" + re.escape(model_name) + r"[\s(:]",
        re.MULTILINE,
    )
    class_m = class_def_re.search(source)
    if not class_m:
        return ApplyResult(
            rule.rev_id, False,
            f"クラス '{model_name}' が {rule.target} に見つかりません",
        )

    # クラス本体の範囲（開始〜次のクラス定義 or EOF）
    next_class_m = re.search(r"\n^class\s+", source[class_m.start() + 1:], re.MULTILINE)
    class_body_end = (class_m.start() + 1 + next_class_m.start()) if next_class_m else len(source)
    class_body = source[class_m.start(): class_body_end]

    # 冪等チェック
    if re.search(r"^\s+" + re.escape(field_name) + r"\s*[:=]", class_body, re.MULTILINE):
        return ApplyResult(
            rule.rev_id, True,
            f"フィールド '{field_name}' は '{model_name}' に既に存在します（冪等スキップ）",
        )

    new_source = source

    # Optional が必要なら import を確認・追加
    if "Optional" in field_type and "Optional" not in source:
        typing_m = re.search(r"^(from typing import\s+)(.+)$", new_source, re.MULTILINE)
        if typing_m:
            imports = sorted({x.strip() for x in typing_m.group(2).split(",")}) + ["Optional"]
            new_source = (
                new_source[: typing_m.start()]
                + typing_m.group(1)
                + ", ".join(sorted(set(imports)))
                + new_source[typing_m.end():]
            )
        else:
            new_source = "from typing import Optional\n" + new_source

    # Field が必要なら import を確認・追加
    use_field = bool(description)
    if use_field and "Field" not in new_source:
        pydantic_m = re.search(r"^(from pydantic import\s+)(.+)$", new_source, re.MULTILINE)
        if pydantic_m:
            imports = sorted({x.strip() for x in pydantic_m.group(2).split(",")} | {"Field"})
            new_source = (
                new_source[: pydantic_m.start()]
                + pydantic_m.group(1)
                + ", ".join(imports)
                + new_source[pydantic_m.end():]
            )

    # フィールド定義行を構築
    if use_field:
        field_line = (
            f'    {field_name}: {field_type} = Field(default={default}, description="{description}")\n'
        )
    else:
        field_line = f"    {field_name}: {field_type} = {default}\n"

    # クラス末尾（最後の非空行の直後）に挿入
    lines = new_source.splitlines(keepends=True)
    class_start_line = next(
        i for i, ln in enumerate(lines)
        if re.match(r"^class\s+" + re.escape(model_name) + r"[\s(:]", ln)
    )
    next_class_line = next(
        (i for i in range(class_start_line + 1, len(lines)) if re.match(r"^class\s+", lines[i])),
        len(lines),
    )
    last_content = class_start_line
    for i in range(class_start_line + 1, next_class_line):
        if lines[i].strip():
            last_content = i

    lines.insert(last_content + 1, field_line)
    target_path.write_text("".join(lines), encoding="utf-8")

    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"'{model_name}' に '{field_name}: {field_type}' を追加しました",
        changed_file=rule.target,
        diff_summary=f"+++ {rule.target}\n  追加: {field_line.strip()}",
    )


# ---------------------------------------------------------------------------
# endpoint_add — FastAPI エンドポイント追加
# ---------------------------------------------------------------------------

def _apply_endpoint_add(rule: ImprovementRule) -> ApplyResult:
    """
    指定 Python ファイル（通常 api/main.py）に新規エンドポイントを追加する。

    patch キー:
      method         (str)  HTTP メソッド（GET/POST/PUT/PATCH/DELETE）
      path           (str)  URLパス（"/" 始まり）
      function_name  (str)  Python 関数名
      response_model (str)  レスポンスモデル名（省略時 "dict"）
      body           (str)  関数本体のコード（インデント込み）
    """
    if not rule.target:
        return ApplyResult(rule.rev_id, False, "target が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    method = rule.patch.get("method", "").upper()
    path = rule.patch.get("path", "")
    function_name = rule.patch.get("function_name", "")
    response_model = rule.patch.get("response_model", "dict")
    body = rule.patch.get("body", "    pass")

    # バリデーション
    if method not in _VALID_HTTP_METHODS:
        return ApplyResult(
            rule.rev_id, False,
            f"method が無効です: {method!r}。有効値: {sorted(_VALID_HTTP_METHODS)}",
        )
    if not path.startswith("/"):
        return ApplyResult(rule.rev_id, False, f"path は '/' で始まる必要があります: {path!r}")
    if not function_name or not function_name.isidentifier():
        return ApplyResult(
            rule.rev_id, False,
            f"function_name が Python 識別子として無効です: {function_name!r}",
        )

    target_path = _PROJECT_ROOT / rule.target
    if not target_path.exists():
        return ApplyResult(rule.rev_id, False, f"ファイルが存在しません: {target_path}")

    source = target_path.read_text(encoding="utf-8")

    # 冪等チェック: 同一 method+path のルートが既に存在するか
    route_re = re.compile(
        r'@\w+\.' + method.lower() + r'\s*\(\s*["\']' + re.escape(path) + r'["\']',
    )
    if route_re.search(source):
        return ApplyResult(
            rule.rev_id, True,
            f"パス '{path}' ({method}) は既に {rule.target} に存在します（冪等スキップ）",
        )

    # 関数名の衝突チェック
    fn_re = re.compile(r"^(?:async\s+)?def\s+" + re.escape(function_name) + r"\s*\(", re.MULTILINE)
    if fn_re.search(source):
        return ApplyResult(
            rule.rev_id, False,
            f"関数名 '{function_name}' が {rule.target} に既に存在します（衝突）",
        )

    # デコレーター生成（response_model が "dict" の場合は省略）
    if response_model and response_model != "dict":
        decorator = f'@app.{method.lower()}("{path}", response_model={response_model})'
    else:
        decorator = f'@app.{method.lower()}("{path}")'

    endpoint_code = f"\n\n{decorator}\ndef {function_name}():\n{body}\n"

    target_path.write_text(source.rstrip() + endpoint_code, encoding="utf-8")

    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"エンドポイント '{method} {path}' を {rule.target} に追加しました",
        changed_file=rule.target,
        diff_summary=f"+++ {rule.target}\n{decorator}\ndef {function_name}(): ...",
    )


# ---------------------------------------------------------------------------
# config_value — 設定ファイルの定数値を書き換え
# ---------------------------------------------------------------------------

_PROTECTED_TARGETS_RE = re.compile(r"^lease_intelligence_.*\.py$|^mind\.json$")


def _apply_config_value(rule: ImprovementRule) -> ApplyResult:
    """
    設定ファイル（.py / .env / .toml）内の定数を書き換える。

    match.key  : 変更対象のキー名（必須）
    patch.value: 書き込む新しい値（必須、数値・文字列・bool に対応）
    """
    if not rule.target:
        return ApplyResult(rule.rev_id, False, "target が未指定です")
    if not rule.match:
        return ApplyResult(rule.rev_id, False, "match が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    key = rule.match.get("key", "")
    if not key:
        return ApplyResult(rule.rev_id, False, "match.key が未指定です")
    if "value" not in rule.patch:
        return ApplyResult(rule.rev_id, False, "patch に 'value' キーが必要です")

    new_value = rule.patch["value"]
    target_path = _PROJECT_ROOT / rule.target

    if not target_path.exists():
        return ApplyResult(rule.rev_id, False, f"ファイルが存在しません: {target_path}")

    suffix = target_path.suffix.lower()
    if suffix == ".py":
        return _config_value_py(rule.rev_id, target_path, key, new_value, rule.target)
    elif suffix == ".env":
        return _config_value_env(rule.rev_id, target_path, key, new_value, rule.target)
    elif suffix in (".toml", ".ini", ".cfg"):
        return _config_value_toml(rule.rev_id, target_path, key, new_value, rule.target)
    else:
        return ApplyResult(
            rule.rev_id, False,
            f"未対応の拡張子です: {suffix!r}。対応: .py / .env / .toml / .ini / .cfg",
        )


def _config_value_py(
    rev_id: str, path: Path, key: str, new_value: Any, label: str
) -> ApplyResult:
    source = path.read_text(encoding="utf-8")
    # KEY = value  [# optional comment]
    # 非貪欲な value グループで末尾スペースをコメントグループに渡す
    pattern = re.compile(
        r"^(" + re.escape(key) + r"\s*=\s*)([^#\n]*?)(\s*#[^\n]*)?$",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ApplyResult(rev_id, False, f"キー '{key}' が {label} に見つかりません")

    old_raw = m.group(2).strip()
    comment_part = m.group(3) or ""

    if isinstance(new_value, bool):
        new_raw = "True" if new_value else "False"
    elif isinstance(new_value, str):
        if old_raw.startswith('"'):
            new_raw = f'"{new_value}"'
        elif old_raw.startswith("'"):
            new_raw = f"'{new_value}'"
        else:
            new_raw = f'"{new_value}"'
    else:
        new_raw = str(new_value)

    if old_raw == new_raw:
        return ApplyResult(rev_id, True, f"{key}: 既に {new_raw} です（冪等スキップ）")

    new_source = pattern.sub(lambda mo: mo.group(1) + new_raw + comment_part, source, count=1)
    path.write_text(new_source, encoding="utf-8")

    return ApplyResult(
        rev_id=rev_id,
        success=True,
        message=f"{key}: {old_raw} → {new_raw}",
        changed_file=label,
        diff_summary=f"--- {label}\n  {key}: {old_raw} → {new_raw}",
    )


def _config_value_env(
    rev_id: str, path: Path, key: str, new_value: Any, label: str
) -> ApplyResult:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^(" + re.escape(key) + r"=)(.*)$",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ApplyResult(rev_id, False, f"キー '{key}' が {label} に見つかりません")

    old_raw = m.group(2)
    new_raw = str(new_value)

    if old_raw == new_raw:
        return ApplyResult(rev_id, True, f"{key}: 既に {new_raw!r} です（冪等スキップ）")

    new_source = pattern.sub(lambda mo: mo.group(1) + new_raw, source, count=1)
    path.write_text(new_source, encoding="utf-8")

    return ApplyResult(
        rev_id=rev_id,
        success=True,
        message=f"{key}: {old_raw!r} → {new_raw!r}",
        changed_file=label,
        diff_summary=f"--- {label}\n  {key}: {old_raw!r} → {new_raw!r}",
    )


def _config_value_toml(
    rev_id: str, path: Path, key: str, new_value: Any, label: str
) -> ApplyResult:
    """TOML ファイルのリーフキーを正規表現で置換。コメント・フォーマットを保持する。"""
    source = path.read_text(encoding="utf-8")
    leaf_key = key.split(".")[-1]
    pattern = re.compile(
        r"^(" + re.escape(leaf_key) + r"\s*=\s*)([^\n]*?)(\s*#[^\n]*)?$",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ApplyResult(rev_id, False, f"キー '{key}' が {label} に見つかりません")

    old_raw = m.group(2).strip()
    comment_part = m.group(3) or ""

    if isinstance(new_value, bool):
        new_raw = "true" if new_value else "false"
    elif isinstance(new_value, str):
        new_raw = f'"{new_value}"'
    else:
        new_raw = str(new_value)

    if old_raw == new_raw:
        return ApplyResult(rev_id, True, f"{key}: 既に {new_raw} です（冪等スキップ）")

    new_source = pattern.sub(lambda mo: mo.group(1) + new_raw + comment_part, source, count=1)
    path.write_text(new_source, encoding="utf-8")

    return ApplyResult(
        rev_id=rev_id,
        success=True,
        message=f"{key}: {old_raw} → {new_raw}",
        changed_file=label,
        diff_summary=f"--- {label}\n  {key}: {old_raw} → {new_raw}",
    )


# ---------------------------------------------------------------------------
# llm_diff — LLM（Gemini）経由でコード変更を生成・適用
# ---------------------------------------------------------------------------

def _apply_llm_diff(rule: ImprovementRule) -> ApplyResult:
    """
    自然言語指示を Gemini に渡してコード変更を生成・適用するフォールバックアプライヤー。
    他のタイプで表現しきれない複雑な変更に使う。

    patch.instruction: 変更内容を記述した自然言語指示（必須）
    """
    if not rule.target:
        return ApplyResult(rule.rev_id, False, "target が未指定です")
    if not rule.patch:
        return ApplyResult(rule.rev_id, False, "patch が未指定です")

    instruction = rule.patch.get("instruction", "")
    if not instruction or not instruction.strip():
        return ApplyResult(rule.rev_id, False, "patch.instruction が空です")

    # 保護ファイルチェック: lease_intelligence_*.py / mind.json への自動変更を禁止
    target_basename = Path(rule.target).name
    if _PROTECTED_TARGETS_RE.match(target_basename):
        return ApplyResult(
            rule.rev_id, False,
            f"保護ファイル '{rule.target}' への llm_diff 適用は禁止です（紫苑確認フロー外）",
        )

    target_path = _PROJECT_ROOT / rule.target
    if not target_path.exists():
        return ApplyResult(rule.rev_id, False, f"ファイルが存在しません: {target_path}")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ApplyResult(rule.rev_id, False, "GEMINI_API_KEY が設定されていません")

    current_content = target_path.read_text(encoding="utf-8")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "あなたはPythonコード編集アシスタントです。"
                    "指示に従ってコードを編集し、編集後のファイル全体を返してください。"
                    "説明は不要です。コードのみ返してください。"
                )
            }]
        },
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    f"{instruction}\n\n"
                    f"---\n# 対象ファイル: {rule.target}\n\n"
                    f"```python\n{current_content}\n```"
                )
            }]
        }],
        "generationConfig": {"temperature": 0.1},
    }

    try:
        import httpx as _httpx
        resp = _httpx.post(url, json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return ApplyResult(rule.rev_id, False, f"Gemini API 呼び出しに失敗しました: {e}")

    try:
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        return ApplyResult(rule.rev_id, False, f"Gemini レスポンスのパースに失敗しました: {e}")

    # コードブロック（```python ... ``` または ``` ... ```）を抽出
    code_block_re = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    code_m = code_block_re.search(generated_text)
    new_content = code_m.group(1) if code_m else generated_text.strip()

    if not new_content:
        return ApplyResult(rule.rev_id, False, "Gemini レスポンスからコードを抽出できませんでした")

    target_path.write_text(new_content, encoding="utf-8")

    return ApplyResult(
        rev_id=rule.rev_id,
        success=True,
        message=f"llm_diff applied via Gemini: {rule.target}",
        changed_file=rule.target,
        diff_summary=f"llm_diff: {rule.target} を Gemini 経由で変更しました",
    )
