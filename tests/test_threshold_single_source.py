"""承認ライン（判定ライン）の単一ソース不変条件をロックするガードテスト。

CLAUDE.md: 「承認ラインを参照・複製する箇所は必ず単一ソースを import すること。
ハードコードした別定数を置くと審査結果がモジュールごとに食い違う」。
値そのものは変更せず、定義が 1 箇所（constants.py）に留まり続けることを AST で保証する
（コメント・docstring 中の言及は誤検知しない）。
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NAMES = {"APPROVAL_LINE", "CONDITIONAL_LINE", "REVIEW_LINE"}


def _assigned_names(py_path: Path) -> list[str]:
    """モジュール内で「代入」されている判定ライン名を返す（import は含めない）。"""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Name) and t.id in _NAMES:
                found.append(t.id)
    return found


def test_thresholds_assigned_only_in_constants():
    assigned = _assigned_names(_REPO_ROOT / "constants.py")
    # 3 つとも constants.py でちょうど 1 回ずつ定義される
    assert sorted(assigned) == ["APPROVAL_LINE", "CONDITIONAL_LINE", "REVIEW_LINE"]


def test_scoring_core_does_not_redefine_thresholds():
    # scoring_core は re-export（import）のみで、数値を再代入しない
    assert _assigned_names(_REPO_ROOT / "scoring_core.py") == []
    src = (_REPO_ROOT / "scoring_core.py").read_text(encoding="utf-8")
    assert "from constants import APPROVAL_LINE, CONDITIONAL_LINE, REVIEW_LINE" in src


def test_domain_thresholds_flow_through_single_source():
    # ドメイン契約の Thresholds も constants の値と一致（別勘定の数値を持たない）
    import constants
    from screening_domain.lease_provider import LeaseDomainProvider

    t = LeaseDomainProvider().thresholds()
    assert (t.approval, t.conditional, t.review) == (
        constants.APPROVAL_LINE,
        constants.CONDITIONAL_LINE,
        constants.REVIEW_LINE,
    )
