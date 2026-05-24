"""Walk through a real cache-miss diagnosis.

We have three captured runs of the same agent. They look identical to the human
eye, but the prompt cache hit ratio is sitting around 30%. Run this script to
find out why.
"""

from prompt_pillar import Pillar, find_stable_prefix


SYS_PROMPT_TEMPLATE = (
    "You are a customer support agent for Acme Corp.\n"
    "Be concise. Cite policy IDs.\n"
    "Current time: {ts}\n"
    "Policies: refunds within 30 days, no charges over $500 without manager approval."
)


run_1 = [
    {"role": "system", "content": SYS_PROMPT_TEMPLATE.format(ts="2026-05-24T10:34")},
    {"role": "user", "content": "I want a refund for order 12345"},
]

run_2 = [
    {"role": "system", "content": SYS_PROMPT_TEMPLATE.format(ts="2026-05-24T10:35")},
    {"role": "user", "content": "I want a refund for order 67890"},
]

run_3 = [
    {"role": "system", "content": SYS_PROMPT_TEMPLATE.format(ts="2026-05-24T10:36")},
    {"role": "user", "content": "Where is my order"},
]


def main() -> None:
    runs = [run_1, run_2, run_3]

    print("=== quick prefix scan ===")
    result = find_stable_prefix(runs)
    print(f"stable prefix length: {len(result.stable_prefix_messages)} messages")
    print(f"first divergence at index: {result.first_divergence_index}")
    print(f"estimated cacheable tokens: {result.estimated_cacheable_tokens}")
    print(f"summary: {result.divergence_summary}\n")

    print("=== full diagnostic ===")
    pillar = Pillar(threshold_tokens=50)
    print(pillar.analyze(runs))


if __name__ == "__main__":
    main()
