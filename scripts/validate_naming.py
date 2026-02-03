#!/usr/bin/env python3
"""
Naming convention validator for the event-discovery project.

Checks:
- Python files: snake_case
- Directories: snake_case (excluding hidden dirs)
- Python classes: PascalCase
- Python functions/methods: snake_case
- Python constants (module-level ALL_CAPS): UPPER_SNAKE_CASE

Usage:
    python scripts/validate_naming.py          # Check all
    python scripts/validate_naming.py src/     # Check specific path
    python scripts/validate_naming.py --strict # Fail on warnings too
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
DUNDER = re.compile(r"^__[a-z]+(__[a-z]+)*__$")

IGNORE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".eggs"}
IGNORE_FILES = {"__init__.py", "conftest.py"}


def check_path_naming(root: Path) -> List[Tuple[str, str, str]]:
    """Check file and directory naming conventions."""
    issues = []

    for path in sorted(root.rglob("*")):
        # Skip ignored directories
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        name = path.stem
        if path.is_dir():
            if not SNAKE_CASE.match(name) and not name.startswith("."):
                issues.append(("error", str(path), f"Directory '{name}' is not snake_case"))
        elif path.suffix == ".py" and path.name not in IGNORE_FILES:
            if not SNAKE_CASE.match(name):
                issues.append(("error", str(path), f"Python file '{name}.py' is not snake_case"))

    return issues


def check_python_naming(filepath: Path) -> List[Tuple[str, str, str]]:
    """Check naming conventions inside a Python file using AST."""
    issues = []

    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not PASCAL_CASE.match(node.name):
                issues.append((
                    "error",
                    f"{filepath}:{node.lineno}",
                    f"Class '{node.name}' is not PascalCase",
                ))

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            name = node.name
            if DUNDER.match(name) or name.startswith("_"):
                # Skip dunder methods and private methods (just check the base)
                base = name.lstrip("_")
                if base and not SNAKE_CASE.match(base) and not DUNDER.match(name):
                    issues.append((
                        "warning",
                        f"{filepath}:{node.lineno}",
                        f"Function '{name}' base is not snake_case",
                    ))
            elif not SNAKE_CASE.match(name):
                issues.append((
                    "error",
                    f"{filepath}:{node.lineno}",
                    f"Function '{name}' is not snake_case",
                ))

        elif isinstance(node, ast.Assign):
            # Check module-level constants
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node, ast.Assign):
                    name = target.id
                    # Skip private/dunder
                    if name.startswith("_"):
                        continue
                    # If it looks like it's trying to be a constant (ALL_CAPS pattern with lowercase)
                    if name.isupper() and not UPPER_SNAKE.match(name):
                        issues.append((
                            "warning",
                            f"{filepath}:{node.lineno}",
                            f"Constant '{name}' is not UPPER_SNAKE_CASE",
                        ))

    return issues


def validate(root: Path, strict: bool = False) -> int:
    """Run all validations. Returns exit code (0=pass, 1=fail)."""
    all_issues = []

    # Check file/directory naming
    all_issues.extend(check_path_naming(root))

    # Check Python code naming
    for pyfile in sorted(root.rglob("*.py")):
        if any(part in IGNORE_DIRS for part in pyfile.parts):
            continue
        all_issues.extend(check_python_naming(pyfile))

    # Report
    errors = [i for i in all_issues if i[0] == "error"]
    warnings = [i for i in all_issues if i[0] == "warning"]

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for _, location, message in warnings:
            print(f"  {location}: {message}")

    if errors:
        print(f"\n✗ {len(errors)} error(s):")
        for _, location, message in errors:
            print(f"  {location}: {message}")

    if not errors and not warnings:
        print("✓ All naming conventions pass")
        return 0

    if errors:
        return 1
    if strict and warnings:
        return 1
    return 0


def main():
    strict = "--strict" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    root = Path(args[0]) if args else Path(".")
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(1)

    exit_code = validate(root, strict=strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
