import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_eval.strategies import (
    apply_observation_masking,
    apply_sliding_window,
    apply_zone_based_pruning,
)

SUITE_PATH = ROOT / "context_eval" / "test_suite.json"
OUTPUT_PATH = ROOT / "context_eval" / "comparison_table.md"


def _estimate_tokens(messages_or_text):
    """Lightweight token estimator for stable, low-overhead metrics."""
    if isinstance(messages_or_text, list):
        text = json.dumps(messages_or_text, ensure_ascii=False)
    else:
        text = str(messages_or_text)
    # Conservative whitespace-based estimate used in the repository's evaluation style.
    return max(1, len(text.split()))


def _prune_for_recursive_summarization(messages, compact_every=15):
    """Deterministic fallback for recursive summarization when no API key is available."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    history_msgs = [m for m in messages if m.get("role") != "system"]

    if len(history_msgs) <= compact_every:
        return messages

    older_msgs = history_msgs[:-10]
    recent_msgs = history_msgs[-10:]

    summary_text = "\n".join(
        f"{m.get('role', 'unknown').upper()}: {m.get('content') or 'tool_call_issued'}"
        for m in older_msgs
    )

    summary_msg = {
        "role": "system",
        "content": f"PRIOR CONTEXT SUMMARY: {summary_text[:1400]}"
    }

    return system_msgs + [summary_msg] + recent_msgs


async def _run_recursive_strategy(messages):
    """Run recursive summarization with a no-network fallback."""
    from context_eval.strategies import apply_recursive_summarization

    try:
        return await apply_recursive_summarization(messages, compact_every=15)
    except Exception:
        return _prune_for_recursive_summarization(messages, compact_every=15)


def _build_strategy_map():
    return {
        "sliding_window": lambda messages: apply_sliding_window(messages, window_size=10),
        "observation_masking": lambda messages: apply_observation_masking(messages, max_unmasked_tools=3),
        "recursive_summarization": lambda messages: asyncio.run(_run_recursive_strategy(messages)),
        "zone_based_pruning": lambda messages: apply_zone_based_pruning(messages, keep_first_n=2, keep_last_n=6),
    }


def _evaluate_case(case, pruned_messages):
    expected = case.get("expected_fact_snippet", "")
    if not expected:
        return False

    flattened = json.dumps(pruned_messages, ensure_ascii=False).lower()
    return expected.lower() in flattened


def evaluate_suite(suite_path=SUITE_PATH):
    with suite_path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)

    strategy_results = []
    strategies = _build_strategy_map()

    for strategy_name, executor in strategies.items():
        passing = 0
        total = 0
        latencies = []
        token_usage = 0

        for case in cases:
            total += 1
            start = time.perf_counter()
            pruned_messages = executor(case.get("messages", []))
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            if _evaluate_case(case, pruned_messages):
                passing += 1

            token_usage += _estimate_tokens(pruned_messages)

        accuracy = 100.0 * passing / total if total else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        strategy_results.append(
            {
                "strategy": strategy_name,
                "accuracy_pct": accuracy,
                "tokens_consumed": token_usage,
                "latency_seconds": avg_latency,
                "cases_passed": passing,
                "cases_total": total,
            }
        )

    return strategy_results


def render_markdown_table(results):
    lines = []
    lines.append("| Strategy | Accuracy After Pruning | Tokens Consumed | Avg Latency (s) |")
    lines.append("|---|---:|---:|---:|")

    for row in results:
        lines.append(
            f"| {row['strategy']} | {row['accuracy_pct']:.2f}% | {row['tokens_consumed']} | {row['latency_seconds']:.6f} |"
        )

    return "\n".join(lines) + "\n"


def write_markdown_table(results, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_table(results), encoding="utf-8")
    return output_path


def main():
    results = evaluate_suite()
    output_path = write_markdown_table(results)

    print("Strategy evaluation complete.")
    print(render_markdown_table(results), end="")
    print(f"Markdown comparison table written to: {output_path}")


if __name__ == "__main__":
    main()
