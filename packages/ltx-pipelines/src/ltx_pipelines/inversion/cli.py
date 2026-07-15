from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ContextFlow on LTX-2")
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--source-first-frame", required=True)
    parser.add_argument("--edited-first-frame", required=True)
    parser.add_argument("--source-prompt", default="")
    parser.add_argument("--target-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise SystemExit("ContextFlow CLI wiring is repository-local and should be instantiated from project code.")


if __name__ == "__main__":
    main()
