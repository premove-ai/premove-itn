"""Audit complete candidate tuples against the pre-batch implementation."""

import json
from pathlib import Path

from premove_itn import Candidate, _rust, build_candidate_graph
from premove_itn.candidates import SPAN_KINDS, TOKEN_PATTERN


def reference_graph(text):
    tokens = tuple(TOKEN_PATTERN.finditer(text))
    grouped = {}
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            spoken = text[tokens[start].start() : tokens[end - 1].end()]
            for kind in SPAN_KINDS:
                for replacement in _rust.realize_options(kind.value, spoken):
                    if replacement != spoken:
                        grouped.setdefault((start, end, replacement), set()).add(kind)
    return tuple(
        Candidate(
            start,
            end,
            tokens[start].start(),
            tokens[end - 1].end(),
            text[tokens[start].start() : tokens[end - 1].end()],
            replacement,
            tuple(kind for kind in SPAN_KINDS if kind in kinds),
        )
        for (start, end, replacement), kinds in sorted(grouped.items())
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (root / "eval/voice_agent_itn/voice_agent_eval.jsonl")
        .read_text()
        .splitlines()
    ]
    mismatches = []
    for row in rows:
        if build_candidate_graph(row["text"]) != reference_graph(row["text"]):
            mismatches.append(row["id"])
    print(
        json.dumps(
            {
                "rows": len(rows),
                "exact_graphs_equal": len(rows) - len(mismatches),
                "mismatches": mismatches,
            }
        )
    )
    assert not mismatches
