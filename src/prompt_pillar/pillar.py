"""Pillar: high-level diagnostic over N message runs.

Wraps find_stable_prefix + diff_messages and turns the result into a short,
human-readable report with a recommendation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .prefix import find_stable_prefix, StablePrefixResult
from .diff import diff_messages

Message = dict[str, Any]

# ISO-8601-ish timestamps and UUIDv4. Most common cache-busters in system
# prompts come from one of these two shapes.
_TS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _guess_cache_buster(text_a: str, text_b: str) -> str:
    """Inspect a divergent pair and guess the offending pattern.

    We are deliberately conservative. False positives are worse than no guess.
    """
    if _TS_PATTERN.search(text_a) and _TS_PATTERN.search(text_b):
        return "timestamp"
    if _UUID_PATTERN.search(text_a) and _UUID_PATTERN.search(text_b):
        return "uuid"
    return ""


@dataclass
class PillarReport:
    prefix_result: StablePrefixResult
    reason: str
    recommendation: str
    meets_threshold: bool
    threshold_tokens: int

    def __str__(self) -> str:
        pr = self.prefix_result
        head = (
            f"Stable prefix: {len(pr.stable_prefix_messages)} messages, "
            f"{pr.estimated_cacheable_tokens} estimated tokens"
        )
        threshold_line = (
            f"Threshold: {self.threshold_tokens} tokens "
            f"({'PASS' if self.meets_threshold else 'BELOW'})"
        )
        return "\n".join(
            [
                head,
                threshold_line,
                f"Reason: {self.reason}",
                f"Recommendation: {self.recommendation}",
            ]
        )


class Pillar:
    """High-level cache-stability diagnostic.

    Parameters:
        threshold_tokens: minimum cacheable-token count we expect. If the
            stable prefix is smaller than this, the report flags it.
    """

    def __init__(self, threshold_tokens: int = 1024) -> None:
        self.threshold_tokens = threshold_tokens

    def analyze(self, runs: list[list[Message]]) -> PillarReport:
        pr = find_stable_prefix(runs)

        if not runs or len(runs) < 2:
            return PillarReport(
                prefix_result=pr,
                reason="need at least 2 runs to diagnose a stable prefix",
                recommendation="capture 2 or more real message lists from the same agent and rerun",
                meets_threshold=pr.estimated_cacheable_tokens >= self.threshold_tokens,
                threshold_tokens=self.threshold_tokens,
            )

        diverge_idx = pr.first_divergence_index
        meets = pr.estimated_cacheable_tokens >= self.threshold_tokens

        if meets and "all runs identical" in pr.divergence_summary:
            return PillarReport(
                prefix_result=pr,
                reason="all runs share the full common prefix",
                recommendation="this prompt is already cache-friendly; verify with cachebench",
                meets_threshold=True,
                threshold_tokens=self.threshold_tokens,
            )

        # We have a divergence point inside the shared length range; pull the
        # two diverging messages from run 0 and run 1 to guess the buster.
        reason = pr.divergence_summary
        recommendation = (
            "hoist the non-cacheable content out of the diverging message; "
            "use a separate user-turn header so the shared prefix stays byte-identical"
        )
        if 0 <= diverge_idx < min(len(runs[0]), len(runs[1])):
            d = diff_messages(runs[0][diverge_idx], runs[1][diverge_idx])
            if d.role_changed:
                reason = (
                    f"role changed at index {diverge_idx}: "
                    f"{d.role_a!r} vs {d.role_b!r}"
                )
                recommendation = (
                    "align role ordering across runs; the role sequence itself "
                    "is part of the cache key"
                )
            else:
                content_a = _content_for_msg(runs[0][diverge_idx])
                content_b = _content_for_msg(runs[1][diverge_idx])
                buster = _guess_cache_buster(content_a, content_b)
                if buster == "timestamp":
                    reason = (
                        f"{runs[0][diverge_idx].get('role', '?')} message timestamp "
                        f"injected at index {diverge_idx} (first char diverges at "
                        f"{d.first_char_diverge})"
                    )
                    recommendation = (
                        "move the timestamp out of the cached message; pass it as "
                        "a fresh user-turn block after the cache breakpoint"
                    )
                elif buster == "uuid":
                    reason = (
                        f"{runs[0][diverge_idx].get('role', '?')} message contains "
                        f"a per-call UUID at index {diverge_idx}"
                    )
                    recommendation = (
                        "remove the per-call UUID from cached messages; keep it in a "
                        "downstream user turn or in metadata"
                    )

        return PillarReport(
            prefix_result=pr,
            reason=reason,
            recommendation=recommendation,
            meets_threshold=meets,
            threshold_tokens=self.threshold_tokens,
        )


def _content_for_msg(msg: Message) -> str:
    from .diff import _content_str  # local to avoid leaking helper at top level

    return _content_str(msg)
