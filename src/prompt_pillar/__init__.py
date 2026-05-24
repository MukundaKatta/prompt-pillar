"""prompt-pillar: diagnose why your LLM prompt cache hit ratio is low.

Given N past message lists from the same agent, find the largest stable prefix
that ALL N share. That stable prefix is what should be cacheable. If the prefix
is smaller than you expected, the agent is mutating the prompt prefix between
calls. That is the most common cause of a low cache hit ratio.

Public API:
    find_stable_prefix(runs)  -> StablePrefixResult
    diff_messages(a, b)       -> DiffReport
    Pillar(threshold_tokens)  -> .analyze(runs) -> PillarReport
"""

from .prefix import StablePrefixResult, find_stable_prefix, messages_equal
from .diff import DiffReport, diff_messages
from .pillar import Pillar, PillarReport

__version__ = "0.1.0"

__all__ = [
    "StablePrefixResult",
    "find_stable_prefix",
    "messages_equal",
    "DiffReport",
    "diff_messages",
    "Pillar",
    "PillarReport",
    "__version__",
]
