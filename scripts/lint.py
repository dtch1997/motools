#!/usr/bin/env python
"""Run linting and formatting fixes."""

import subprocess
import sys


def main() -> int:
    """Run ruff check --fix and ruff format."""
    print("Running ruff check --fix...")
    result = subprocess.run(["python", "-m", "ruff", "check", "--fix", "."])
    if result.returncode != 0:
        return result.returncode

    print("\nRunning ruff format...")
    result = subprocess.run(["python", "-m", "ruff", "format", "."])
    if result.returncode != 0:
        return result.returncode

    print("\nLinting and formatting complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
