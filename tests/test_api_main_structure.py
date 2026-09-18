from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_routers_do_not_import_composition_root() -> None:
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "api" / "routers").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "api.main":
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "api.main":
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert violations == [], "router -> api.main reverse imports: " + ", ".join(violations)


def test_external_app_contract_remains_module_level() -> None:
    main_path = REPO_ROOT / "api" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"), filename=str(main_path))
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "app" in assignments


def test_composition_root_size_can_only_ratchet_down() -> None:
    """New behavior belongs in routers/services, not in the composition root."""

    main_path = REPO_ROOT / "api" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(main_path))
    direct_routes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
        and node.func.attr in {"get", "post", "put", "patch", "delete"}
    ]

    assert len(source.splitlines()) <= 8_200
    assert len(direct_routes) <= 21
