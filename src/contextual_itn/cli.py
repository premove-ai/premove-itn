from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextual_itn.generator import generate_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible contextual ITN JSONL data."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for example in generate_examples(args.count, args.seed):
            json.dump(example.to_dict(), output_file, ensure_ascii=False)
            output_file.write("\n")


if __name__ == "__main__":
    main()
