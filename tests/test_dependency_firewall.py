"""decepticons never imports chronohorn. This test enforces it."""
from __future__ import annotations

import ast
from pathlib import Path


def _all_python_files() -> list[Path]:
    src = Path(__file__).resolve().parent.parent / "src" / "decepticons"
    return sorted(src.rglob("*.py"))


def test_no_chronohorn_imports():
    """No source file in decepticons/ may import from chronohorn."""
    violations = []
    for path in _all_python_files():
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "chronohorn" in alias.name:
                        violations.append(f"{path.relative_to(path.parents[3])}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and "chronohorn" in node.module:
                violations.append(f"{path.relative_to(path.parents[3])}:{node.lineno}: from {node.module}")
    assert violations == [], "decepticons must not import chronohorn:\n" + "\n".join(violations)
