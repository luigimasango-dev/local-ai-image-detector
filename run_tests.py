#!/usr/bin/env python3
"""Run every regression suite. No pytest required.

    python run_tests.py

Exits non-zero if anything fails, so it works as a pre-commit / CI check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = sorted((ROOT / "tests").glob("test_*.py"))


def main() -> int:
    if not SUITES:
        print("No test suites found under tests/", file=sys.stderr)
        return 1

    failed = []
    for suite in SUITES:
        print(f"\n{'=' * 60}\n{suite.name}\n{'=' * 60}")
        result = subprocess.run(
            [sys.executable, str(suite)], cwd=str(ROOT), text=True
        )
        if result.returncode != 0:
            failed.append(suite.name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(SUITES)} suite(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
