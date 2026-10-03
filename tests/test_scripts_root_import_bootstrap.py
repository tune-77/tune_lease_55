"""`python scripts/<name>.py` で起動される scripts がリポジトリ直下のモジュールを import できること。

日次パイプラインはこの形で起動するため sys.path[0] は scripts/ になる。リポジトリ直下を
sys.path に入れないまま try の中で api/ などを import すると、ModuleNotFoundError を握りつぶして
黙って処理を飛ばす（Cloud Run の個人記憶・判断状態イベントの取り込みが止まっていた）。
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT_PACKAGES = {"api", "scripts", "scoring", "utils", "screening_domain", "memory_layers", "evaluators", "mebuki", "components"}
BOOT = re.compile(r"sys\.path\.(insert|append)\(|site\.addsitedir\(")


def _root_imports(tree: ast.AST, root_modules: set[str]) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            name = node.module
        elif isinstance(node, ast.Import):
            name = node.names[0].name
        else:
            continue
        if name.split(".")[0] in root_modules:
            out.append((node.lineno, name))
    return out


def test_scripts_importing_repo_modules_add_repo_root_to_sys_path():
    root_modules = {p.stem for p in REPO.glob("*.py")} | ROOT_PACKAGES
    missing = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        imports = _root_imports(ast.parse(text), root_modules)
        if imports and not BOOT.search(text):
            missing.append(f"{path.name}: {imports[0][1]}")
    assert not missing, "リポジトリ直下を sys.path に入れずに import している: " + ", ".join(missing)


def test_silent_failure_import_comes_after_bootstrap():
    late = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        at = text.find("from silent_failure_log import")
        if at < 0:
            continue
        boot = BOOT.search(text)
        if boot is None or boot.start() > at:
            late.append(path.name)
    assert not late, "sys.path 追加より前に silent_failure_log を import している: " + ", ".join(late)
