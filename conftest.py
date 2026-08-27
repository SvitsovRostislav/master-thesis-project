"""Repo-root pytest conftest: make `src` importable without an installed package.

Mirrors the `sys.path.insert(0, str(REPO_ROOT))` pattern already used by
every script in scripts/ -- there is no setup.py/pyproject.toml install
step for this project, so tests need the same explicit path setup.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
