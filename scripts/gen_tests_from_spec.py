"""
P0-004: SPECファイルからpytestテストスケルトンを自動生成する。
AC-xxx Given-When-Then を読み取り、1:1対応するテスト関数を出力する。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

_SPEC_ID_RE = re.compile(r"^P\d+-\d{3}$")
_AC_RE = re.compile(r"^\s*-\s*(?:\*\*)?(AC-\d{3})(?:\*\*)?\s*[:：]?\s*(.+)$")
_AC_HEADER_RE = re.compile(r"^#{1,6}\s+.*[Aa]cceptance\s+[Cc]riteria", re.MULTILINE)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pytest skeleton from SPEC Acceptance Criteria"
    )
    parser.add_argument("spec_path", metavar="SPEC_PATH", help="Path to SPEC markdown")
    parser.add_argument("--out-dir", default="tests", help="Test output root (default: tests)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing")
    return parser.parse_args(argv)


def load_spec(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    end = text.index("---", 3)
    fm_text = text[3:end].strip()
    body = text[end + 3:].strip()
    fm = yaml.safe_load(fm_text) or {}
    return fm, body


def validate_frontmatter(fm: dict) -> tuple[str, int]:
    if "spec_id" not in fm:
        print("invalid frontmatter: missing spec_id", file=sys.stderr)
        sys.exit(3)
    if "phase" not in fm:
        print("invalid frontmatter: missing phase", file=sys.stderr)
        sys.exit(3)
    spec_id = str(fm["spec_id"])
    if not _SPEC_ID_RE.match(spec_id):
        print(f"invalid spec_id format: {spec_id!r} (expected P<N>-<NNN>)", file=sys.stderr)
        sys.exit(3)
    phase = int(fm["phase"])
    return spec_id, phase


def extract_acceptance_criteria(body: str) -> list[tuple[str, str]]:
    m = _AC_HEADER_RE.search(body)
    if m is None:
        print("Acceptance Criteria section not found", file=sys.stderr)
        sys.exit(4)
    section = body[m.end():]
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for line in section.splitlines():
        match = _AC_RE.match(line)
        if match:
            ac_id = match.group(1)
            text = match.group(2).strip()
            if ac_id in seen:
                print(f"warning: duplicate {ac_id}, keeping first", file=sys.stderr)
                continue
            seen.add(ac_id)
            results.append((ac_id, text))
    return results


def _escape_docstring(text: str) -> str:
    return text.replace('"""', r'\"\"\"')


def render_test_file(spec_id: str, phase: int, acs: list[tuple[str, str]]) -> str:
    lines: list[str] = [
        '"""',
        f"Auto-generated test skeleton for {spec_id}.",
        "DO NOT EDIT the AC docstrings manually — regenerate via:",
        f"    python scripts/gen_tests_from_spec.py specs/phase{phase}/{spec_id}-*.md",
        "Each test_ac_xxx corresponds 1:1 with AC-xxx in the SPEC.",
        '"""',
        "import pytest",
        "",
        f'SPEC_ID = "{spec_id}"',
        f"PHASE = {phase}",
        "",
    ]
    if not acs:
        lines.append("# No AC entries found in SPEC")
    for ac_id, text in acs:
        num = ac_id.replace("AC-", "").zfill(3)
        func_name = f"test_ac_{num}"
        docstring = _escape_docstring(f"{ac_id}: {text}")
        lines += [
            "",
            f"def {func_name}() -> None:",
            f'    """',
            f"    {docstring}",
            f'    """',
            f'    pytest.fail("{ac_id} not implemented")',
        ]
    lines.append("")
    return "\n".join(lines)


def write_output(
    out_path: Path,
    content: str,
    force: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would write to {out_path}")
        print(content)
        return
    if out_path.exists() and not force:
        print(f"output exists: {out_path} (use --force)", file=sys.stderr)
        sys.exit(5)
    out_path.write_text(content, encoding="utf-8")


def ensure_init_files(out_dir: Path, phase_dir: Path) -> None:
    for d in (out_dir, phase_dir):
        init = d / "__init__.py"
        if not init.exists():
            init.touch()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    spec_path = Path(args.spec_path)

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(2)

    fm, body = load_spec(spec_path)
    spec_id, phase = validate_frontmatter(fm)
    print(f"[gen_tests] parsed spec: {spec_id} (phase={phase})")

    acs = extract_acceptance_criteria(body)
    print(f"[gen_tests] extracted {len(acs)} AC entries")

    out_dir = Path(args.out_dir)
    phase_dir = out_dir / f"spec_phase{phase}"
    out_path = phase_dir / f"test_{spec_id}.py"

    if not args.dry_run:
        phase_dir.mkdir(parents=True, exist_ok=True)
        ensure_init_files(out_dir, phase_dir)

    content = render_test_file(spec_id, phase, acs)
    print(f"[gen_tests] writing {out_path}")
    write_output(out_path, content, force=args.force, dry_run=args.dry_run)

    print("[gen_tests] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
