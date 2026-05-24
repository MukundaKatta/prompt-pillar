"""CLI: `python -m prompt_pillar <jsonl-of-message-lists>`.

Input file format: one JSON value per line, where each line is a list of
messages (one full run). At least 2 lines are needed for a useful diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .pillar import Pillar


def _load_runs(path: Path) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"line {line_no}: invalid JSON ({exc})", file=sys.stderr)
            sys.exit(2)
        if not isinstance(value, list):
            print(f"line {line_no}: expected a list of messages", file=sys.stderr)
            sys.exit(2)
        runs.append(value)
    return runs


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m prompt_pillar <jsonl-of-message-lists> [threshold]")
        print("  each line of the file is one full run (a JSON list of messages)")
        return 0

    path = Path(args[0])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    threshold = 1024
    if len(args) > 1:
        try:
            threshold = int(args[1])
        except ValueError:
            print(f"threshold must be an int, got {args[1]!r}", file=sys.stderr)
            return 2

    runs = _load_runs(path)
    pillar = Pillar(threshold_tokens=threshold)
    report = pillar.analyze(runs)
    print(report)
    return 0 if report.meets_threshold else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
