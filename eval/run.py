"""Thin wrapper: `python eval/run.py ...` is `resolver eval ...`."""

import sys

from resolver.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
