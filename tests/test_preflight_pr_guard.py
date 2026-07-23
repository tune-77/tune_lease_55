"""scripts/preflight_pr_guard.py の純粋関数を git 非依存で検証する。

雛形: tests/test_no_duplicate_toplevel_defs.py（AST を機械的に検査するテスト）。
git diff 収集層（subprocess）は対象外とし、AST / Import / Circuit Breaker の
判定ロジックだけをユニットテストする。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = REPO_ROOT / "scripts" / "preflight_pr_guard.py"

_spec = importlib.util.spec_from_file_location("preflight_pr_guard", _MODULE_PATH)
assert _spec and _spec.loader
g = importlib.util.module_from_spec(_spec)
# dataclass の文字列アノテーション解決に sys.modules 登録が必要
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)


# ── Import Sanitizer ─────────────────────────────────────────────────────────

def test_declared_dependencies_include_known_packages():
    declared = g.load_declared_dependencies()
    # pyproject / requirements の宣言が正規化名で拾えていること
    assert "scikit-learn" in declared
    assert "pyyaml" in declared
    assert "fastapi" in declared


def test_parse_requirement_names_strips_versions_and_comments():
    text = "streamlit==1.54.0\n# comment\npandas==2.3.0\n\nflask>=3.0.0  # inline\n-r other.txt\n"
    names = g.parse_requirement_names(text)
    assert names == {"streamlit", "pandas", "flask"}


def test_resolve_import_accepts_stdlib_alias_and_local():
    declared = g.load_declared_dependencies()
    local = g.local_top_level_modules()
    assert g.resolve_import("os", False, declared, local) is True
    assert g.resolve_import("json", False, declared, local) is True
    # 別名（import 名 sklearn ⇄ 配布名 scikit-learn）/ ローカルモジュール
    assert g.resolve_import("sklearn", False, declared, local) is True
    assert g.resolve_import("scoring_core", False, declared, local) is True
    # 相対 import はローカル前提で常に許容
    assert g.resolve_import("", True, declared, local) is True


def test_resolve_import_flags_hallucinated_package():
    declared = g.load_declared_dependencies()
    local = g.local_top_level_modules()
    assert g.resolve_import("nonexistent_pkg_xyz", False, declared, local) is False


def test_import_sanitizer_warns_only_on_unknown_imports():
    diff = (
        "+import os\n"
        "+import pandas as pd\n"
        "+from .relative_mod import thing\n"
        "+import nonexistent_pkg_xyz\n"
        "+from fake_module_abc import stuff\n"
    )
    warnings = g.run_import_sanitizer(diff)
    flagged = {w.detail for w in warnings}
    assert "import nonexistent_pkg_xyz" in flagged
    assert "from fake_module_abc import stuff" in flagged
    # 正当な import は警告しない
    assert not any("import os" == w.detail for w in warnings)
    assert not any("import pandas as pd" == w.detail for w in warnings)


def test_extract_added_imports_ignores_diff_header_and_handles_forms():
    diff = (
        "+++ b/foo.py\n"       # ヘッダは無視
        "+import a, b as c\n"
        "+from x.y import z\n"
        "+from . import rel\n"
        "-import removed_mod\n"  # 削除行は対象外
    )
    got = g.extract_added_imports(diff)
    tops = [t[0] for t in got]
    assert "a" in tops and "b" in tops   # カンマ / as を分解
    assert "x" in tops                   # from x.y -> top-level x
    assert not any(t[0] == "removed_mod" for t in got)
    assert any(t[2] for t in got)        # 相対 import が is_relative=True で拾える


# ── AST Guard ────────────────────────────────────────────────────────────────

def test_check_ast_syntax_detects_syntax_error():
    assert g.check_ast_syntax("def f(:\n    pass", "bad.py") is not None
    assert g.check_ast_syntax("def f():\n    return 1", "good.py") is None


def test_sanity_check_detects_line_drop():
    original = "\n".join(f"line_{i} = {i}" for i in range(20))
    new = "line_0 = 0"
    w = g.sanity_check(original, new)
    assert w is not None and "行数" in w.message


def test_sanity_check_detects_function_loss():
    # 行数は 50% 以上維持しつつ関数だけ 3 つ以上消えるケース
    # （行数激減チェックが先に発火しないよう filler を足す）
    original = "".join(f"def f{i}():\n    pass\n" for i in range(6))  # 12 行 / 6 関数
    new = "def f0():\n    pass\ndef f1():\n    pass\n" + "".join(
        f"# filler {i}\n" for i in range(6)
    )  # 10 行 / 2 関数（4 関数消失）
    w = g.sanity_check(original, new)
    assert w is not None and "関数" in w.message


def test_sanity_check_passes_normal_edit():
    original = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    new = "def a():\n    return 10\n\ndef b():\n    return 2\n"
    assert g.sanity_check(original, new) is None


# ── Circuit Breaker ──────────────────────────────────────────────────────────

def test_evaluate_retry_resets_when_clean():
    assert g.evaluate_retry(3, False, 2) == (0, False)


def test_evaluate_retry_trips_after_max():
    # 上限 2: count が 3 になった時点で発火
    assert g.evaluate_retry(0, True, 2) == (1, False)
    assert g.evaluate_retry(1, True, 2) == (2, False)
    assert g.evaluate_retry(2, True, 2) == (3, True)


def test_run_circuit_breaker_uses_state_file(tmp_path):
    state = tmp_path / "retries.json"
    sig = "deadbeef"
    # 警告ありで上限まで積む → 3 回目で発火
    assert g.run_circuit_breaker(sig, True, 2, state) is None
    assert g.run_circuit_breaker(sig, True, 2, state) is None
    tripped = g.run_circuit_breaker(sig, True, 2, state)
    assert tripped is not None and tripped.guard == "circuit_breaker"
    # 警告が解消したらリセットされ、発火しない
    assert g.run_circuit_breaker(sig, False, 2, state) is None
