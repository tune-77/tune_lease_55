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


# ---------- 裸の数値リテラルによる判定分岐の検出 ----------
#
# 上の3テストは「定数名の再代入」しか見ないため、`if total >= 70:` のような
# 数値直書きは素通りしていた。実際 lease_intelligence_tools.get_score_detail が
# `>= 70` を直書きしており、APPROVAL_LINE=71 とずれてスコア70の案件が
# ツールによって「承認」にも「条件付き承認」にもなる状態だった。

_VERDICT_WORDS = {"承認", "条件付き承認", "条件付承認", "否決", "要審議"}


def _verdict_strings(node: ast.AST) -> set[str]:
    """その分岐で判定を「決めている」文字列だけを返す。

    `final == "承認"` のような比較は判定の決定ではないため対象外にする
    （代入・return されているものだけを見る）。
    """
    out: set[str] = set()
    for sub in ast.walk(node):
        values: list[ast.AST] = []
        if isinstance(sub, ast.Assign):
            values = [sub.value]
        elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
            values = [sub.value]
        elif isinstance(sub, ast.Return) and sub.value is not None:
            values = [sub.value]
        for value in values:
            for leaf in ast.walk(value):
                if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                    if leaf.value.strip() in _VERDICT_WORDS:
                        out.add(leaf.value.strip())
    return out


# 判定ラインとして意味を持つ値のみ対象にする。物件スコアの帯（80/50/30）や
# 件数比較（0/1）まで拾うと誤検知だらけになり、ガードが無視される。
def _threshold_values() -> set[float]:
    import constants

    values = {
        float(constants.APPROVAL_LINE),
        float(constants.CONDITIONAL_LINE),
        float(getattr(constants, "REVIEW_LINE", constants.CONDITIONAL_LINE)),
        70.0,  # APPROVAL_LINE=71 に対する典型的な取り違え値
    }
    return values


# 判定ラインを直書きしていることが分かっている箇所。値は「なぜ直せないか」の理由。
_VERDICT_LITERAL_ALLOWLIST = {
    "mobile_app/api.py": (
        "めぶき簡易審査は別サーフェスで、条件付ラインが 50（CONDITIONAL_LINE=60 と不一致）。"
        "揃えると簡易審査の判定出力が変わるため、プロダクト判断が要る"
    ),
}


def _numeric_verdict_branches(py_path: Path) -> list[str]:
    """判定ライン相当の数値リテラルで判定文字列を決めている箇所を返す。"""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    targets = _threshold_values()
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        literals = [
            float(c.value)
            for c in node.test.comparators
            if isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
            and not isinstance(c.value, bool)
        ]
        if not any(v in targets for v in literals):
            continue
        verdicts: set[str] = set()
        for b in list(node.body) + list(node.orelse):
            verdicts |= _verdict_strings(b)
        if verdicts:
            rel = py_path.relative_to(_REPO_ROOT)
            hits.append(f"{rel}:{node.lineno} 数値{literals} → {sorted(verdicts)}")
    return hits


def test_no_numeric_literal_decides_a_verdict():
    """判定（承認/条件付き承認/否決）を数値リテラルで分岐しないこと。

    閾値は constants.APPROVAL_LINE / CONDITIONAL_LINE を import して使う。
    """
    skip_parts = {".git", "node_modules", "_archive", ".venv", "__pycache__", "tests"}
    hits: list[str] = []
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        if any(part in skip_parts for part in path.parts):
            continue
        if str(path.relative_to(_REPO_ROOT)) in _VERDICT_LITERAL_ALLOWLIST:
            continue
        hits.extend(_numeric_verdict_branches(path))

    assert hits == [], (
        "判定を数値リテラルで分岐している箇所があります。"
        "constants.APPROVAL_LINE / CONDITIONAL_LINE を使ってください:\n  "
        + "\n  ".join(hits)
    )
