# prompt-pillar

Diagnose why your LLM prompt cache hit ratio is low.

Given N past message lists from the same agent, `prompt-pillar` finds the largest stable prefix that every run shares. That stable prefix is what should be cacheable. If the prefix is smaller than you expected, something in your agent is mutating the prompt prefix between calls. That is the most common cause of a low cache hit ratio.

Zero runtime dependencies. Python 3.10+.

## Install

```bash
pip install prompt-pillar
```

## Use it

```python
from prompt_pillar import find_stable_prefix, diff_messages, Pillar

result = find_stable_prefix([msgs_run1, msgs_run2, msgs_run3])
print(result.stable_prefix_messages)   # list[dict]: the safe-to-cache prefix
print(result.first_divergence_index)   # int: index of the first message that differs
print(result.divergence_summary)       # str: human-readable summary
print(result.estimated_cacheable_tokens)

# Per-message pairwise diff with first-character divergence index
diff_messages(msgs_run1[2], msgs_run2[2])

# Full diagnostic report
pillar = Pillar(threshold_tokens=1024)
print(pillar.analyze([msgs_run1, msgs_run2, msgs_run3]))
```

Sample output:

```
Stable prefix: 0 messages, 0 estimated tokens
Threshold: 50 tokens (BELOW)
Reason: system message timestamp injected at index 0 (first char diverges at 73)
Recommendation: move the timestamp out of the cached message; pass it as a fresh user-turn block after the cache breakpoint
```

## CLI

```bash
python -m prompt_pillar examples/sample.jsonl
```

Each line of the file is one full run, encoded as a JSON list of messages.

## Recognized message shapes

Anthropic, OpenAI, and plain `{role, content}` dicts. Dict key order does not matter; two messages that serialize to the same canonical JSON are considered equal.

## Where this sits

- [cachebench](https://github.com/MukundaKatta/cachebench) measures the cache hit ratio
- [prompt-cache-warmer](https://github.com/MukundaKatta/prompt-cache-warmer) pre-warms the cache
- [llm-message-hash](https://crates.io/crates/llm-message-hash) computes canonical hashes
- `prompt-pillar` (this lib) diagnoses why the ratio is low in the first place

## License

MIT
