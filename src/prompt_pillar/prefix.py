"""Find the longest stable prefix shared across N message lists.

A "message" is a dict with at least `role` and `content`. We accept the common
shapes: Anthropic (role/content), OpenAI (role/content + optional name), and
plain dicts. Equality is order-insensitive for dict keys so a run that serialized
keys differently still matches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


Message = dict[str, Any]


def _canon(value: Any) -> str:
    """Canonical JSON of a value with sorted keys.

    Used for equality so two messages that differ only in dict key order still
    compare equal. This matters because some SDKs reorder keys on serialize.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def messages_equal(a: Message, b: Message) -> bool:
    """Two messages are equal when their canonical JSON is byte-identical."""
    return _canon(a) == _canon(b)


def _estimate_tokens(messages: list[Message]) -> int:
    """Char-based token heuristic: 1 token ~ 4 chars. Good enough for triage."""
    total_chars = sum(len(_canon(m)) for m in messages)
    return total_chars // 4


def _describe_message(msg: Message) -> str:
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, str):
        snippet = content[:40].replace("\n", " ")
        return f"{role}: {snippet!r}"
    return f"{role}: <structured>"


@dataclass
class StablePrefixResult:
    stable_prefix_messages: list[Message] = field(default_factory=list)
    first_divergence_index: int = 0
    divergence_summary: str = ""
    estimated_cacheable_tokens: int = 0
    run_count: int = 0


def find_stable_prefix(runs: list[list[Message]]) -> StablePrefixResult:
    """Return the longest message prefix that is byte-identical across every run.

    If there is only one run, the whole run is its own stable prefix.
    If runs is empty, the prefix is empty.
    """
    if not runs:
        return StablePrefixResult(
            stable_prefix_messages=[],
            first_divergence_index=0,
            divergence_summary="no runs supplied",
            estimated_cacheable_tokens=0,
            run_count=0,
        )

    if len(runs) == 1:
        only = list(runs[0])
        return StablePrefixResult(
            stable_prefix_messages=only,
            first_divergence_index=len(only),
            divergence_summary="only one run; nothing to compare",
            estimated_cacheable_tokens=_estimate_tokens(only),
            run_count=1,
        )

    base = runs[0]
    min_len = min(len(r) for r in runs)
    prefix_len = 0
    diverge_summary = ""

    for i in range(min_len):
        base_msg = base[i]
        all_equal = True
        for run_idx in range(1, len(runs)):
            other = runs[run_idx][i]
            if not messages_equal(base_msg, other):
                all_equal = False
                diverge_summary = (
                    f"message index {i} differs between run 0 and run {run_idx}: "
                    f"run 0 = [{_describe_message(base_msg)}], "
                    f"run {run_idx} = [{_describe_message(other)}]"
                )
                break
        if not all_equal:
            break
        prefix_len = i + 1

    # If we hit min_len with full agreement but runs differ in length, the
    # shorter run is itself the divergence point.
    if prefix_len == min_len and not all(len(r) == min_len for r in runs):
        diverge_summary = (
            f"all {min_len} shared messages match, but runs have different "
            f"lengths {[len(r) for r in runs]}"
        )

    prefix = list(base[:prefix_len])
    return StablePrefixResult(
        stable_prefix_messages=prefix,
        first_divergence_index=prefix_len,
        divergence_summary=diverge_summary or "all runs identical for shared length",
        estimated_cacheable_tokens=_estimate_tokens(prefix),
        run_count=len(runs),
    )
