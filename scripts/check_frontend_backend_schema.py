#!/usr/bin/env python3
"""フロントエンド型定義とバックエンドPydanticスキーマの配線整合性を検査する。

検査のみで、コードは書き換えない。

背景:
    `api/schemas.py` の `ScoringRequest` と `frontend/src/types/index.ts` の
    `ScoringFormData` は、審査フォームの入力契約をフィールド名で1:1対応させる
    ことで型を揃えている（CLAUDE.md「数値単位（バグの温床）」参照）。
    どちらか一方にだけフィールドを足すと、審査データが静かに欠落するか、
    フロントが送っていない値をバックエンドが参照する。

    さらに、金額系フィールドはフロント入力が百万円、バックエンドが千円のため
    `frontend/src/lib/scoringUnits.ts` の `monetaryScoringFields` に載っている
    フィールドだけが ×1000 変換される。ここへの追加漏れは実例のある事故
    （2026-07レビューで承認ライン二重定義が見つかった話と同種の「単一ソース
    崩れ」）で、金額が桁違いのままスコアリングされても例外を出さない。

使い方:
    python3 scripts/check_frontend_backend_schema.py

終了コード:
    0 = error なし（warning は許容）
    1 = error あり
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCHEMAS_PATH = REPO_ROOT / "api" / "schemas.py"
DEFAULT_TYPES_PATH = REPO_ROOT / "frontend" / "src" / "types" / "index.ts"
DEFAULT_UNITS_PATH = REPO_ROOT / "frontend" / "src" / "lib" / "scoringUnits.ts"

PYDANTIC_MODEL = "ScoringRequest"
TS_INTERFACE = "ScoringFormData"

# ScoringFormData にはあるが ScoringRequest には意図的に無いフィールド。
# 送信前にドロップされるUI専用値、または別経路で送る値。
_FRONTEND_ONLY_FIELDS: dict[str, str] = {
    "trend_grade_t0": "AI分析用の表示専用フィールド（今期格付）。送信payloadには含めない",
    "trend_grade_t1": "AI分析用の表示専用フィールド（前期格付）。送信payloadには含めない",
    "trend_grade_t2": "AI分析用の表示専用フィールド（前々期格付）。送信payloadには含めない",
    "acceptance_year": "UI表示専用（検収年）。ScoringRequestには送らない",
    "strength_tags": "定性評価の強みタグ。passion_textとは別経路で扱うUI専用フィールド",
}

# ScoringRequest の float フィールドのうち、百万円/千円の金額ではないため
# monetaryScoringFields（×1000変換対象）に含めなくてよいもの。
_NON_MONETARY_FLOAT_FIELDS: dict[str, str] = {
    "competitor_rate": "金利（%）であり金額ではない",
    "asset_score": "0-100の物件スコアであり金額ではない",
}

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class Finding:
    level: str
    check: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "check": self.check, "message": self.message}


def parse_pydantic_model_fields(path: Path, model_name: str) -> dict[str, str]:
    """AST で Pydantic モデルのフィールド名 -> 型注釈文字列 を抽出する。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == model_name:
            fields: dict[str, str] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields[stmt.target.id] = ast.unparse(stmt.annotation)
            return fields
    raise ValueError(f"{path} に class {model_name} が見つかりません")


def parse_ts_interface_fields(path: Path, interface_name: str) -> dict[str, str]:
    """正規表現で TS interface のフィールド名 -> 型文字列 を抽出する。"""
    text = path.read_text(encoding="utf-8")
    block_match = re.search(
        r"export\s+interface\s+" + re.escape(interface_name) + r"\s*\{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    if not block_match:
        raise ValueError(f"{path} に interface {interface_name} が見つかりません")

    fields: dict[str, str] = {}
    for line in block_match.group(1).splitlines():
        stripped = line.split("//", 1)[0].strip()
        if not stripped:
            continue
        field_match = re.match(r"(\w+)\??:\s*([^;]+);?$", stripped)
        if field_match:
            fields[field_match.group(1)] = field_match.group(2).strip()
    return fields


def parse_ts_string_array(path: Path, const_name: str) -> list[str]:
    """正規表現で `export const <name> = [...]` の文字列リテラル一覧を抽出する。"""
    text = path.read_text(encoding="utf-8")
    array_match = re.search(
        r"export\s+const\s+" + re.escape(const_name) + r"\s*=\s*\[(.*?)\]",
        text,
        re.DOTALL,
    )
    if not array_match:
        raise ValueError(f"{path} に const {const_name} が見つかりません")
    return re.findall(r'"([^"]+)"', array_match.group(1))


def check_field_parity(
    request_fields: dict[str, str], form_fields: dict[str, str]
) -> list[Finding]:
    findings: list[Finding] = []

    backend_only = set(request_fields) - set(form_fields)
    for name in sorted(backend_only):
        findings.append(
            Finding(
                ERROR,
                "missing_in_frontend",
                f"{PYDANTIC_MODEL}.{name} が {TS_INTERFACE} にありません。"
                "フロントから値が送られず審査データが欠落します"
                "（意図的なら frontend/src/types/index.ts に追加するか、"
                "このスクリプトの _FRONTEND_ONLY_FIELDS の逆パターンを検討してください）",
            )
        )

    frontend_only = set(form_fields) - set(request_fields)
    unexplained_frontend_only = frontend_only - set(_FRONTEND_ONLY_FIELDS)
    for name in sorted(unexplained_frontend_only):
        findings.append(
            Finding(
                ERROR,
                "missing_in_backend",
                f"{TS_INTERFACE}.{name} が {PYDANTIC_MODEL} にありません。"
                "バックエンドに届かない値です"
                "（意図的なUI専用フィールドなら _FRONTEND_ONLY_FIELDS に理由付きで追加してください）",
            )
        )

    stale_allowlist = set(_FRONTEND_ONLY_FIELDS) - frontend_only
    for name in sorted(stale_allowlist):
        findings.append(
            Finding(
                WARNING,
                "stale_allowlist",
                f"_FRONTEND_ONLY_FIELDS の {name} は {TS_INTERFACE} に存在しません。"
                "削除されたフィールドの残骸の可能性があります",
            )
        )

    return findings


def check_monetary_unit_conversion(
    request_fields: dict[str, str],
    form_fields: dict[str, str],
    monetary_fields: list[str],
) -> list[Finding]:
    findings: list[Finding] = []
    monetary_set = set(monetary_fields)

    float_fields_in_payload = {
        name
        for name, annotation in request_fields.items()
        if "float" in annotation and name in form_fields
    }

    missing_conversion = float_fields_in_payload - monetary_set - set(_NON_MONETARY_FLOAT_FIELDS)
    for name in sorted(missing_conversion):
        findings.append(
            Finding(
                ERROR,
                "unit_conversion_missing",
                f"{PYDANTIC_MODEL}.{name} はfloatでフォームにも存在しますが、"
                "monetaryScoringFields（百万円→千円の×1000変換対象）に含まれていません。"
                "金額フィールドなら scoringUnits.ts の monetaryScoringFields に追加、"
                "金額でないなら _NON_MONETARY_FLOAT_FIELDS に理由付きで追加してください",
            )
        )

    stale_monetary = monetary_set - float_fields_in_payload
    for name in sorted(stale_monetary):
        findings.append(
            Finding(
                WARNING,
                "unit_conversion_extra",
                f"monetaryScoringFields の {name} は {PYDANTIC_MODEL} のfloatフィールドとして"
                "見つかりません（削除・リネームされた可能性があります）",
            )
        )

    stale_non_monetary_allowlist = set(_NON_MONETARY_FLOAT_FIELDS) - float_fields_in_payload
    for name in sorted(stale_non_monetary_allowlist):
        findings.append(
            Finding(
                WARNING,
                "stale_allowlist",
                f"_NON_MONETARY_FLOAT_FIELDS の {name} は {PYDANTIC_MODEL} のfloatフィールドとして"
                "見つかりません。削除されたフィールドの残骸の可能性があります",
            )
        )

    return findings


def run_checks(
    schemas_path: Path = DEFAULT_SCHEMAS_PATH,
    types_path: Path = DEFAULT_TYPES_PATH,
    units_path: Path = DEFAULT_UNITS_PATH,
) -> dict[str, Any]:
    request_fields = parse_pydantic_model_fields(schemas_path, PYDANTIC_MODEL)
    form_fields = parse_ts_interface_fields(types_path, TS_INTERFACE)
    monetary_fields = parse_ts_string_array(units_path, "monetaryScoringFields")

    findings: list[Finding] = []
    findings.extend(check_field_parity(request_fields, form_fields))
    findings.extend(check_monetary_unit_conversion(request_fields, form_fields, monetary_fields))

    counts = {
        level: sum(1 for f in findings if f.level == level)
        for level in (ERROR, WARNING, INFO)
    }
    return {
        "backend_field_count": len(request_fields),
        "frontend_field_count": len(form_fields),
        "monetary_field_count": len(monetary_fields),
        "counts": counts,
        "status": "error" if counts[ERROR] else ("warning" if counts[WARNING] else "ok"),
        "findings": [f.as_dict() for f in findings],
    }


def format_report(report: dict[str, Any]) -> str:
    icons = {ERROR: "❌", WARNING: "⚠️ ", INFO: "ℹ️ "}
    lines = [
        f"フロント/バックエンド スキーマ整合性チェック（{PYDANTIC_MODEL} ⇔ {TS_INTERFACE}）",
        f"  backend fields={report['backend_field_count']}"
        f"  frontend fields={report['frontend_field_count']}"
        f"  monetary fields={report['monetary_field_count']}",
        f"  総合: {report['status'].upper()}"
        f"  error={report['counts']['error']}"
        f" warning={report['counts']['warning']}",
        "",
    ]
    for finding in report["findings"]:
        lines.append(f"{icons.get(finding['level'], '')} [{finding['check']}] {finding['message']}")
    if not report["findings"]:
        lines.append("✅ 指摘なし")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemas", type=Path, default=DEFAULT_SCHEMAS_PATH)
    parser.add_argument("--types", type=Path, default=DEFAULT_TYPES_PATH)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument("--json", type=Path, help="レポートの JSON 出力先")
    args = parser.parse_args(argv)

    report = run_checks(args.schemas, args.types, args.units)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON: {args.json}")

    return 1 if report["counts"][ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
