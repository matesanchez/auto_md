from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIR_NAMES = {".pytest_cache", ".ruff_cache", "__pycache__"}
FILE_NAMES = {".DS_Store"}
SEARCH_ROOTS = ["automd", "tests", "scripts", "reports", "."]
SKIP_DIRS = {".git", "external", "runs", ".venv", "venv"}


def iter_cleanup_paths():
    seen: set[Path] = set()
    for name in SEARCH_ROOTS:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path in seen:
                continue
            seen.add(path)
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            yield path


def main() -> int:
    removed = 0
    for path in iter_cleanup_paths():
        if path.name in DIR_NAMES and path.is_dir():
            shutil.rmtree(path)
            removed += 1
        elif path.name in FILE_NAMES and path.is_file():
            path.unlink()
            removed += 1
    print(f"removed {removed} generated cache files/directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
