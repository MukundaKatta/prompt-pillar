"""Pairwise diff of two messages with character-level granularity."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


def _content_str(msg: Message) -> str:
    """Coerce message content to a string for diffing.

    Anthropic structured content (list of blocks) gets rendered as JSON-ish
    text so the diff still shows where the divergence is.
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    # Structured content: render with stable repr so diff is deterministic.
    import json

    return json.dumps(content, sort_keys=True, ensure_ascii=False, indent=2)


def _first_char_diverge(a: str, b: str) -> int:
    """Return the index of the first differing character, or len(min(a,b))."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class DiffReport:
    role_changed: bool
    role_a: str
    role_b: str
    content_unified_diff: str
    first_char_diverge: int
    identical: bool


def diff_messages(a: Message, b: Message) -> DiffReport:
    """Diff two messages. Returns role change, char-diverge index, unified diff."""
    role_a = str(a.get("role", ""))
    role_b = str(b.get("role", ""))
    content_a = _content_str(a)
    content_b = _content_str(b)

    identical = role_a == role_b and content_a == content_b
    if identical:
        return DiffReport(
            role_changed=False,
            role_a=role_a,
            role_b=role_b,
            content_unified_diff="",
            first_char_diverge=len(content_a),
            identical=True,
        )

    unified = "\n".join(
        difflib.unified_diff(
            content_a.splitlines(),
            content_b.splitlines(),
            fromfile="a",
            tofile="b",
            lineterm="",
        )
    )
    return DiffReport(
        role_changed=role_a != role_b,
        role_a=role_a,
        role_b=role_b,
        content_unified_diff=unified,
        first_char_diverge=_first_char_diverge(content_a, content_b),
        identical=False,
    )
