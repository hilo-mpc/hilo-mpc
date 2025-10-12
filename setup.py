#!/usr/bin/env python3
"""
Legacy setuptools shim.

This project now uses Poetry for builds and packaging.
To build or install locally, use:
  - poetry install
  - poetry build

Keeping this file avoids breakage for tools that still probe for setup.py,
but it will not execute a real build.
"""

import sys

sys.stderr.write(
    "This project uses Poetry. Use 'poetry install' for development or 'poetry build' to create distributions.\n"
)
raise SystemExit(1)
