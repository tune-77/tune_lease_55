"""承認ライン値の数値リテラル直書きをラチェットで締めるガードテスト。

test_threshold_single_source.py の `test_no_numeric_literal_decides_a_verdict` は
「判定文字列を決める `if` 比較」しか見ないため、次の形が素通りしていた:

    _eff_approval = 71                        # 代入
    def f(threshold: float = 71.0): ...       # デフォルト引数
    res.get("approval_line", 71)              # get のフォールバック
    scoreToAngle(71) / f"閾値71"              # 表示・テンプレート

いずれも constants.APPROVAL_LINE を env で変えたときに追従せず、審査結果や
画面表示がモジュールごとに食い違う。CLAUDE.md が禁じている二重定義と同型。

方針: APPROVAL_LINE と同値の数値リテラルを AST で数え、ファイル単位の
既知件数を上限として固定する。新規ファイルでの直書きも、既知ファイルでの
件数増加も ERROR にする。直書きを解消したら件数を減らす（ラチェットを締める）。
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 走査対象外。_archive は退役コード、pydeps はベンダー(numpy)、
# tests は本ガード自身が値を書くため除く。
_SKIP_PARTS = {
    ".git", ".claude", "node_modules", "_archive", ".venv",
    "__pycache__", "pydeps", "tests", "frontend", "docs",
}

# 既知の直書き件数。値は (件数, 理由)。
#
# 「無関係」= APPROVAL_LINE とは別文脈でたまたま同じ数値になっているもの。
#   これは恒久的な許可であり、解消の必要はない。
# 「未解消」= 本来 APPROVAL_LINE を import すべき負債。別PRで順次潰す。
_KNOWN_LITERALS: dict[str, tuple[int, str]] = {
    # ---- 無関係（恒久許可）----
    "api/context/sentiment_enricher.py": (
        1, "建設業のセンチメント指数 71.0。承認ラインとは無関係"
    ),
    "quantum_analysis_module.py": (
        1, "日本標準産業分類コードの範囲 71〜79。承認ラインとは無関係"
    ),
    # ---- 未解消の負債 ----
    "lease_logic_sumaho8.py": (
        4, "実行経路のないデッドコード。ファイルごと削除する方針（別PR）"
    ),
    "mobile_app/advisor_strategy.py": (4, "未解消: めぶき側の判定分岐"),
    "mobile_app/api.py": (3, "未解消: めぶき簡易審査API"),
    "soul_factor_miner.py": (3, "未解消: threshold = 71.0 の直書き3箇所"),
    "components/dashboard.py": (3, "未解消: 閾値最適化の参照点 abs(th - 71)"),
    "mathematician_agent.py": (2, "未解消: 参照点 REF と教師ラベル生成"),
    "charts.py": (2, "未解消: ゲージの色帯境界"),
    "analysis_regression.py": (1, "未解消: current_line = 71"),
    "knowledge.py": (1, "未解消: approval_line の表示フォールバック"),
    "scripts/append_sample_cases.py": (1, "未解消: サンプル生成の判定"),
    "visual_insights.py": (1, "未解消: 参照線 add_vline(x=71)"),
}


def _count_approval_literals(py_path: Path, value: float) -> int:
    """ファイル内に現れる承認ライン同値の数値リテラル数を返す。"""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return 0
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            continue
        if float(node.value) == value:
            count += 1
    return count


def _scan() -> dict[str, int]:
    import constants

    value = float(constants.APPROVAL_LINE)
    found: dict[str, int] = defaultdict(int)
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        n = _count_approval_literals(path, value)
        if n:
            found[str(path.relative_to(_REPO_ROOT))] = n
    return dict(found)


def test_no_new_approval_line_literal():
    """未登録ファイルでの新規直書き・既知ファイルでの件数増加を禁止する。"""
    found = _scan()
    problems: list[str] = []
    for rel, count in sorted(found.items()):
        allowed, _reason = _KNOWN_LITERALS.get(rel, (0, ""))
        if count > allowed:
            problems.append(f"{rel}: {count}件（許可 {allowed}件）")

    assert problems == [], (
        "承認ライン値の数値リテラル直書きが増えています。"
        "constants.APPROVAL_LINE を import してください:\n  "
        + "\n  ".join(problems)
    )


def test_allowlist_has_no_stale_entry():
    """直書きを解消したのに許可リストへ残していないこと（ラチェットを緩めない）。

    件数が減ったらこのテストが落ちるので、_KNOWN_LITERALS の数値を
    実際の件数まで下げる（0 になったらエントリごと消す）。
    """
    found = _scan()
    stale: list[str] = []
    for rel, (allowed, _reason) in sorted(_KNOWN_LITERALS.items()):
        actual = found.get(rel, 0)
        if actual < allowed:
            stale.append(f"{rel}: 実際 {actual}件 < 許可 {allowed}件")

    assert stale == [], (
        "直書きが減ったのに許可リストの件数が古いままです。"
        "_KNOWN_LITERALS を実件数まで下げてください:\n  "
        + "\n  ".join(stale)
    )


def test_fixed_files_stay_clean():
    """本PRで単一ソース化したファイルに直書きが戻っていないこと。

    ここに挙げたファイルは許可リストに載せない（載せた時点でガードが緩む）。
    """
    fixed = [
        "components/score_calculation.py",
        "components/score_gauge.py",
        "components/batch_scoring.py",
        "components/model_diagnostics_view.py",
        "web/export_visualization_data.py",
    ]
    found = _scan()
    regressed = [f"{f}: {found[f]}件" for f in fixed if found.get(f)]
    assert regressed == [], (
        "単一ソース化済みのファイルに直書きが再混入しています:\n  "
        + "\n  ".join(regressed)
    )
