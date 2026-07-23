#!/usr/bin/env python3
"""PR 発行直前のプリフライト検証ガード（AST / Import Sanitizer / Circuit Breaker）。

AI が生成したコードに稀に混入する事故を「最後の一線」で検知する。

現段階の方針は **警告のみ（warn-only）**：
  - 既定では警告を出しても exit 0（PR 発行を止めない）。段階導入の可視化フェーズ。
  - 将来ブロック化する場合は ``--strict`` を付ける（警告があれば exit 1）。

3 つのガード：
  1. AST Guard    — 変更 .py の構文チェック（py_compile 相当）＋行数激減/関数消失検知。
     出典: ``.agents/skills/auto-improvement-pipeline/scripts/step3_auto_apply.py``
           の ``Step3AutoApplier._run_local_tests`` / ``_sanity_check``。当該モジュールは
           import が重いため、同等の小ロジックを本モジュール内に再実装している。
  2. Import Sanitizer — 差分の追加行に現れた新規 import を stdlib / 宣言済み依存 /
     ローカルモジュール / 現環境（find_spec）で解決し、どれにも当たらなければ
     「幻覚 import の疑い」を警告。宣言済み依存の出典は ``pyproject.toml`` /
     ``requirements.txt`` / ``web/requirements.txt``。副次的に pyflakes があれば併用。
  3. Circuit Breaker — 同一ファイル集合に対し警告が解消しないまま繰り返された回数を
     ``.claude/state/preflight_retries.json`` に記録し、既定回数を超えたら人間への
     バトンタッチを警告。env 命名は ``scripts/execute_codex_queue.py`` の
     ``CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES``（既定 2）規約に整合させ、
     ``PREFLIGHT_MAX_RETRIES``（既定 2）とする。

CLI:
    python3 scripts/preflight_pr_guard.py            # 人間可読レポート（warn-only）
    python3 scripts/preflight_pr_guard.py --json     # 機械可読
    python3 scripts/preflight_pr_guard.py --strict   # 警告があれば exit 1
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

# circuit breaker の既定値。execute_codex_queue.py の
# CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES（既定 2）と同じ規約に揃える。
DEFAULT_MAX_RETRIES = 2
RETRY_STATE_PATH = REPO_ROOT / ".claude" / "state" / "preflight_retries.json"

# import 名 ⇄ 配布名（PyPI 名）が食い違う代表例の別名表。
# ここに無くても find_spec フォールバックで実在すれば通るため、
# 誤検知を減らすための「宣言済み依存とのマッチ補助」に留める。
IMPORT_TO_DIST_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "psycopg2": "psycopg2-binary",
    "google": "google-genai",  # google 名前空間はプレフィックス一致でも許容する
}

# google など、複数の配布パッケージが相乗りする名前空間ルート。
# top-level 名がこれらの場合は、宣言済み依存に "<root>-*" があれば解決とみなす。
NAMESPACE_ROOTS = {"google"}


@dataclass
class Warning_:
    """1 件の警告。guard は 'ast' / 'import' / 'circuit_breaker' のいずれか。"""

    guard: str
    message: str
    file: str | None = None
    detail: str | None = None


@dataclass
class GuardReport:
    warnings: list[Warning_] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    base_ref: str | None = None

    def add(self, w: Warning_) -> None:
        self.warnings.append(w)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# git 収集層
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: Path = REPO_ROOT) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:  # noqa: BLE001 — git 不在等でも落とさず警告なしにする
        return 1, "", str(e)


def resolve_base_ref() -> str | None:
    """比較基準の ref を決める。origin/master → master → origin/main → main の順。

    どれも解決できなければ None（その場合は HEAD 未コミット差分にフォールバック）。
    """
    for ref in ("origin/master", "master", "origin/main", "main"):
        code, _, _ = _run_git(["rev-parse", "--verify", "--quiet", ref])
        if code == 0:
            # マージベースがあればそちらを使う（分岐点からの差分に限定）
            mb_code, mb_out, _ = _run_git(["merge-base", "HEAD", ref])
            if mb_code == 0 and mb_out.strip():
                return mb_out.strip()
            return ref
    return None


def git_changed_py_files(base_ref: str | None) -> list[Path]:
    """base_ref（無ければ HEAD）から見た変更 .py ファイル（作業ツリー込み）。"""
    diff_args = ["diff", "--name-only"]
    diff_args.append(base_ref if base_ref else "HEAD")
    code, out, _ = _run_git(diff_args)
    files: list[Path] = []
    if code == 0:
        for line in out.splitlines():
            name = line.strip()
            if name.endswith(".py"):
                p = REPO_ROOT / name
                if p.is_file():
                    files.append(p)
    return files


def git_diff_text(base_ref: str | None) -> str:
    """base_ref（無ければ HEAD）からの unified diff テキスト。"""
    code, out, _ = _run_git(["diff", base_ref if base_ref else "HEAD"])
    return out if code == 0 else ""


def git_show_original(base_ref: str | None, path: Path) -> str | None:
    """base_ref 時点の内容を取得。新規ファイル等で取得できなければ None。"""
    if not base_ref:
        base_ref = "HEAD"
    rel = path.relative_to(REPO_ROOT).as_posix()
    code, out, _ = _run_git(["show", f"{base_ref}:{rel}"])
    return out if code == 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Guard 1: AST（構文・健全性）
# ─────────────────────────────────────────────────────────────────────────────

def check_ast_syntax(source: str, path_label: str) -> Warning_ | None:
    """構文チェック（py_compile 相当）。SyntaxError なら警告を返す。"""
    try:
        compile(source, path_label, "exec")
    except SyntaxError as e:
        return Warning_(
            guard="ast",
            message="構文エラー（SyntaxError）を検知しました",
            file=path_label,
            detail=f"{e.msg} (line {e.lineno})",
        )
    return None


def sanity_check(original: str, new: str) -> Warning_ | None:
    """行数激減・関数消失を検知する。

    出典: step3_auto_apply.py の Step3AutoApplier._sanity_check の再実装。
    しきい値も同じ（行数 <50%、関数消失 >2）。
    """
    orig_lines = len(original.splitlines())
    new_lines = len(new.splitlines())
    if orig_lines > 0 and new_lines < orig_lines * 0.5:
        return Warning_(
            guard="ast",
            message="行数が激減しています（意図しない削除の可能性）",
            detail=f"{orig_lines} → {new_lines} 行",
        )
    try:
        orig_tree = ast.parse(original)
        new_tree = ast.parse(new)
    except SyntaxError:
        # 構文エラーは check_ast_syntax 側で報告するのでここでは触らない
        return None

    orig_funcs = {n.name for n in ast.walk(orig_tree) if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef)}
    lost = orig_funcs - new_funcs
    if len(lost) > 2:
        return Warning_(
            guard="ast",
            message="関数が複数消失しています（意図しない削除の可能性）",
            detail=f"消失: {sorted(lost)}",
        )
    return None


def run_ast_guard(base_ref: str | None, files: Iterable[Path]) -> list[Warning_]:
    warnings: list[Warning_] = []
    for path in files:
        label = path.relative_to(REPO_ROOT).as_posix()
        try:
            new_src = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            warnings.append(Warning_("ast", "ファイル読み取り失敗", label, str(e)))
            continue

        syn = check_ast_syntax(new_src, label)
        if syn:
            warnings.append(syn)
            continue  # 構文が壊れている場合、健全性チェックは無意味

        original = git_show_original(base_ref, path)
        if original is not None:
            sc = sanity_check(original, new_src)
            if sc:
                sc.file = label
                warnings.append(sc)
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Guard 2: Import Sanitizer
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_dist(name: str) -> str:
    """PyPI 名 / import 名を正規化（小文字化・区切りをハイフンに統一）。"""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def parse_requirement_names(text: str) -> set[str]:
    """requirements.txt 形式のテキストから正規化済み配布名を抽出する。"""
    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # extras / version 指定を落として素の名前だけ取る
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(_normalize_dist(m.group(1)))
    return names


def _parse_pyproject_deps(text: str) -> set[str]:
    """pyproject.toml の [project].dependencies と optional-dependencies から名前を抽出。

    tomllib 依存を避けるため、"..." で囲まれた依存指定行を正規表現で拾う軽量パース。
    dependencies / optional-dependencies いずれの配列にも対応。
    """
    names: set[str] = set()
    for m in re.finditer(r'"([A-Za-z0-9_.\-]+(?:\[[^\]]*\])?[^"]*)"', text):
        spec = m.group(1)
        name = re.match(r"^([A-Za-z0-9_.\-]+)", spec)
        # バージョン記号や比較演算子を含む依存指定っぽい行だけを対象化しつつ、
        # 素の名前（timesfm 等ピン無し）も拾う
        if name:
            names.add(_normalize_dist(name.group(1)))
    return names


def load_declared_dependencies(repo_root: Path = REPO_ROOT) -> set[str]:
    """pyproject.toml / requirements.txt / web/requirements.txt の宣言済み依存を集約。"""
    declared: set[str] = set()
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        # dependencies と optional-dependencies のブロックだけを対象にする
        text = pyproject.read_text(encoding="utf-8")
        dep_blocks = re.findall(
            r"(?:^|\n)\s*(?:dependencies|dev|[A-Za-z0-9_-]+)\s*=\s*\[(.*?)\]",
            text,
            flags=re.DOTALL,
        )
        for block in dep_blocks:
            declared |= _parse_pyproject_deps(block)
    for req in ("requirements.txt", "web/requirements.txt"):
        p = repo_root / req
        if p.is_file():
            declared |= parse_requirement_names(p.read_text(encoding="utf-8"))
    return declared


def local_top_level_modules(repo_root: Path = REPO_ROOT) -> set[str]:
    """リポジトリ直下の .py / パッケージディレクトリ名（＝ローカル import 元）。"""
    mods: set[str] = set()
    for entry in repo_root.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix == ".py":
            mods.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            mods.add(entry.name)
        elif entry.is_dir():
            # __init__.py が無くても import 経路になりうる主要ディレクトリを拾う
            if any(entry.glob("*.py")):
                mods.add(entry.name)
    return mods


def extract_added_imports(diff_text: str) -> list[tuple[str, str, bool]]:
    """unified diff の追加行から (top_level_module, raw_line, is_relative) を抽出。

    差分ヘッダ（+++）は除外する。相対 import（from . / from .. ）は is_relative=True。
    """
    results: list[tuple[str, str, bool]] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        code = line[1:].strip()
        # from X import ...
        m = re.match(r"^from\s+(\.*)([A-Za-z0-9_.]*)\s+import\s+", code)
        if m:
            dots, mod = m.group(1), m.group(2)
            if dots:  # 相対 import
                results.append((mod.split(".")[0] if mod else "", code, True))
            elif mod:
                results.append((mod.split(".", 1)[0], code, False))
            continue
        # import X[, Y] / import X as Z
        m = re.match(r"^import\s+(.+)$", code)
        if m:
            for part in m.group(1).split(","):
                name = part.strip().split(" as ")[0].strip()
                if name:
                    results.append((name.split(".")[0], code, False))
    return results


def _dist_matches(top_level: str, declared: set[str]) -> bool:
    norm = _normalize_dist(top_level)
    if norm in declared:
        return True
    alias = IMPORT_TO_DIST_ALIASES.get(top_level)
    if alias and _normalize_dist(alias) in declared:
        return True
    # google 等の名前空間ルートは "<root>-*" があれば解決とみなす
    if top_level in NAMESPACE_ROOTS:
        if any(d.startswith(f"{norm}-") for d in declared):
            return True
    return False


def resolve_import(
    top_level: str,
    is_relative: bool,
    declared: set[str],
    local_mods: set[str],
) -> bool:
    """import が解決可能なら True。どれにも当たらなければ False（幻覚の疑い）。"""
    if is_relative:
        return True  # 相対 import はローカル前提。ここでは幻覚判定の対象外
    if not top_level:
        return True
    if top_level in sys.stdlib_module_names:
        return True
    if top_level in local_mods:
        return True
    if _dist_matches(top_level, declared):
        return True
    # フォールバック: 現環境に実在するか（推移依存・別名の誤検知抑制）
    try:
        if importlib.util.find_spec(top_level) is not None:
            return True
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    return False


def run_import_sanitizer(diff_text: str, repo_root: Path = REPO_ROOT) -> list[Warning_]:
    warnings: list[Warning_] = []
    declared = load_declared_dependencies(repo_root)
    local_mods = local_top_level_modules(repo_root)
    seen: set[str] = set()
    for top_level, raw_line, is_relative in extract_added_imports(diff_text):
        key = f"{top_level}:{raw_line}"
        if key in seen:
            continue
        seen.add(key)
        if not resolve_import(top_level, is_relative, declared, local_mods):
            warnings.append(
                Warning_(
                    guard="import",
                    message=f"幻覚 import の疑い: '{top_level}' は stdlib / 宣言済み依存 / "
                    "ローカルモジュール / 現環境のいずれにも見つかりません",
                    detail=raw_line,
                )
            )
    return warnings


def run_pyflakes(files: list[Path]) -> list[Warning_]:
    """pyflakes があれば未使用 import / 未定義名を副次的に警告（任意・非致命）。"""
    if importlib.util.find_spec("pyflakes") is None or not files:
        return []
    warnings: list[Warning_] = []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pyflakes", *[str(f) for f in files]],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return []
    for line in (proc.stdout + proc.stderr).splitlines():
        low = line.lower()
        if "imported but unused" in low or "undefined name" in low:
            warnings.append(Warning_(guard="import", message="pyflakes", detail=line.strip()))
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Guard 3: Circuit Breaker（リトライ計数）
# ─────────────────────────────────────────────────────────────────────────────

def changed_files_signature(files: Iterable[Path], branch: str) -> str:
    rels = sorted(p.relative_to(REPO_ROOT).as_posix() for p in files)
    payload = branch + "|" + "|".join(rels)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def evaluate_retry(prev_count: int, has_warnings: bool, max_retries: int) -> tuple[int, bool]:
    """新しい試行回数と、ブレーカー発火（人間へバトンタッチ警告）要否を返す。

    - 警告が無ければカウントをリセット（0）。
    - 警告があればカウント +1。max_retries を超えたら発火。
    """
    if not has_warnings:
        return 0, False
    new_count = prev_count + 1
    return new_count, new_count > max_retries


def _load_retry_state(state_path: Path) -> dict:
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_retry_state(state_path: Path, state: dict) -> None:
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # 状態保存の失敗でプリフライトを止めない


def run_circuit_breaker(
    signature: str, has_warnings: bool, max_retries: int, state_path: Path = RETRY_STATE_PATH
) -> Warning_ | None:
    state = _load_retry_state(state_path)
    entry = state.get(signature, {})
    prev_count = int(entry.get("count", 0))
    new_count, tripped = evaluate_retry(prev_count, has_warnings, max_retries)
    state[signature] = {"count": new_count}
    _save_retry_state(state_path, state)
    if tripped:
        return Warning_(
            guard="circuit_breaker",
            message=f"同一箇所への修正が {new_count} 回連続で警告解消に至っていません。"
            f"自律リトライ枠（上限 {max_retries}）を超えました。人間へバトンタッチを推奨します。",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# オーケストレーション
# ─────────────────────────────────────────────────────────────────────────────

def run_preflight(max_retries: int | None = None) -> GuardReport:
    if max_retries is None:
        max_retries = int(os.environ.get("PREFLIGHT_MAX_RETRIES", DEFAULT_MAX_RETRIES))

    base_ref = resolve_base_ref()
    files = git_changed_py_files(base_ref)
    diff_text = git_diff_text(base_ref)

    report = GuardReport(
        checked_files=[p.relative_to(REPO_ROOT).as_posix() for p in files],
        base_ref=base_ref,
    )

    for w in run_ast_guard(base_ref, files):
        report.add(w)
    for w in run_import_sanitizer(diff_text):
        report.add(w)
    for w in run_pyflakes(files):
        report.add(w)

    # circuit breaker は「コード品質の警告（ast/import）」が解消したかで判定する。
    quality_warnings = any(w.guard in ("ast", "import") for w in report.warnings)
    _, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    signature = changed_files_signature(files, branch.strip() or "HEAD")
    breaker = run_circuit_breaker(signature, quality_warnings, max_retries)
    if breaker:
        report.add(breaker)

    return report


def format_human(report: GuardReport) -> str:
    lines: list[str] = []
    lines.append("── プリフライト検証ガード（警告のみ） ─────────────")
    lines.append(f"基準 ref: {report.base_ref or 'HEAD (未コミット差分)'}")
    lines.append(f"検査対象 .py: {len(report.checked_files)} 件")
    if not report.warnings:
        lines.append("✅ 警告なし")
        return "\n".join(lines)
    lines.append(f"⚠️  警告 {len(report.warnings)} 件:")
    label = {"ast": "AST", "import": "Import", "circuit_breaker": "CircuitBreaker"}
    for w in report.warnings:
        loc = f" [{w.file}]" if w.file else ""
        lines.append(f"  • ({label.get(w.guard, w.guard)}){loc} {w.message}")
        if w.detail:
            lines.append(f"      ↳ {w.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR 前プリフライト検証ガード（警告のみ）")
    parser.add_argument("--json", action="store_true", help="機械可読 JSON で出力")
    parser.add_argument(
        "--strict", action="store_true", help="警告があれば exit 1（既定は warn-only の exit 0）"
    )
    parser.add_argument(
        "--max-retries", type=int, default=None,
        help=f"circuit breaker のリトライ上限（既定 {DEFAULT_MAX_RETRIES} / env PREFLIGHT_MAX_RETRIES）",
    )
    args = parser.parse_args(argv)

    report = run_preflight(max_retries=args.max_retries)

    if args.json:
        print(json.dumps(
            {
                "base_ref": report.base_ref,
                "checked_files": report.checked_files,
                "warnings": [asdict(w) for w in report.warnings],
                "has_warnings": report.has_warnings,
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print(format_human(report))

    if args.strict and report.has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
