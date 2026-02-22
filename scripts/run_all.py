#!/usr/bin/env python3
"""CLI entry point for the BioNN benchmark suite."""

from __future__ import annotations

import argparse
import logging
import sys

from bionn.config import load_config
from bionn.runner import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="BioNN Benchmark Suite")
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to YAML config override (merged on top of config/default.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    run_suite(cfg)


if __name__ == "__main__":
    main()
