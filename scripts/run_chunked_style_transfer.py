#!/usr/bin/env python3
"""Entry point for the chunked reference-guided style transfer pipeline.

See scripts/chunked_style_transfer_ltx23/ for the package implementation and
chunked_style_transfer_plan.md for the design this implements.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chunked_style_transfer_ltx23.cli import main

if __name__ == "__main__":
    main()
