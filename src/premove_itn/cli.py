"""Command-line interface for contextual inverse text normalization."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version

from .contextual import SUPPORTED_DEVICES, PremoveITN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="premove-itn",
        description=(
            "Contextual inverse text normalization for English voice-agent transcripts."
        ),
        epilog=(
            "If text is omitted, newline-delimited transcripts are read from stdin."
        ),
    )
    parser.add_argument("text", nargs="?", help="transcript to normalize")
    parser.add_argument(
        "--device",
        choices=sorted(SUPPORTED_DEVICES),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('premove-itn')}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.text is None and sys.stdin.isatty():
        parser.error(
            "no input provided; pass TEXT or pipe newline-delimited text to stdin"
        )

    try:
        itn = PremoveITN.from_pretrained(device=args.device)
    except Exception as exc:
        print(f"premove-itn: failed to load model: {exc}", file=sys.stderr)
        return 1

    inputs = (
        [args.text]
        if args.text is not None
        else (line.rstrip("\r\n") for line in sys.stdin)
    )
    for text in inputs:
        try:
            output = itn.normalize(text)
        except Exception as exc:
            print(f"premove-itn: failed to normalize input: {exc}", file=sys.stderr)
            return 1
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
