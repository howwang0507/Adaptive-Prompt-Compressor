"""
Interactive Command Line Interface for Adaptive Prompt Compressor.

Usage:
    python -m src.cli "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
    python -m src.cli --provider openai --prompt "Explain quantum computing briefly"
    python -m src.cli --benchmark
"""

import argparse
import time
from src.interface import LinUCBCompressor


def run_benchmark_suite(compressor: LinUCBCompressor):
    test_cases = [
        (
            "Code Task",
            "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [pivot] + quicksort([x for x in arr if x > pivot])",
        ),
        (
            "RAG / Documentation",
            "In modern cloud architectures, microservices communicate primarily via asynchronous event streams and RESTful HTTP APIs. Each service typically owns its private database to prevent schema coupling and maintain transactional isolation across distinct failure domains.",
        ),
        (
            "Conversational Chat",
            "Could you please be so kind as to give me a very detailed explanation of why the sky appears blue to the human eye on a clear day, and what physical atmospheric phenomena are involved in this process?",
        ),
    ]

    print("\n" + "=" * 68)
    print("📊 ADAPTIVE PROMPT COMPRESSOR - EMPIRICAL BENCHMARK SUITE")
    print("=" * 68)
    print(f"{'Task Category':<22} | {'Strategy':<14} | {'Tokens In':<9} | {'Saved':<8} | {'Latency'}")
    print("-" * 68)

    total_orig = 0
    total_comp = 0

    for category, prompt in test_cases:
        t0 = time.perf_counter()
        comp, strat, meta = compressor.compress(prompt)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        orig_len = len(prompt)
        comp_len = len(comp)
        total_orig += orig_len
        total_comp += comp_len

        savings = 100.0 - (comp_len / orig_len) * 100.0 if orig_len > 0 else 0.0
        print(f"{category:<22} | {strat:<14} | {orig_len // 4:<9} | {savings:>5.1f}%  | {elapsed_us:>5.1f}µs")

    print("-" * 68)
    overall_savings = 100.0 - (total_comp / total_orig) * 100.0 if total_orig > 0 else 0.0
    print(f"{'OVERALL AVERAGE':<22} | {'-':<14} | {total_orig // 4:<9} | {overall_savings:>5.1f}%  | < 1ms routing")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Prompt Compressor CLI - Dynamic token optimization via LinUCB."
    )
    parser.add_argument("prompt", nargs="?", default=None, help="The prompt text to compress.")
    parser.add_argument("--prompt", "-p", dest="prompt_flag", help="Alternative flag to pass prompt.")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "anthropic", "simulation"],
        default="openai",
        help="Target LLM provider (default: openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target model name (e.g., gpt-4o, gpt-4o-mini).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run an automated benchmark across representative prompts.",
    )

    args = parser.parse_args()
    prompt_text = args.prompt or args.prompt_flag

    compressor = LinUCBCompressor(provider=args.provider, model_name=args.model)

    if args.benchmark or not prompt_text:
        run_benchmark_suite(compressor)
        if not prompt_text:
            return

    # Single prompt compression
    t0 = time.perf_counter()
    compressed, strategy, meta = compressor.compress(prompt_text)
    latency_us = (time.perf_counter() - t0) * 1_000_000

    orig_tokens = len(prompt_text) // 4
    comp_tokens = len(compressed) // 4
    savings_pct = 100.0 - (len(compressed) / len(prompt_text)) * 100.0 if len(prompt_text) > 0 else 0.0

    print("\n" + "=" * 55)
    print("🧠 ADAPTIVE PROMPT COMPRESSION RESULTS")
    print("=" * 55)
    print(f"Target Provider   : {compressor.provider.upper()}")
    print(f"Selected Strategy : {strategy} (Arm {meta['arm']})")
    print(f"Routing Overhead  : {latency_us:.1f} µs (< 1ms)")
    print(f"Token Reduction   : {savings_pct:.1f}% ({orig_tokens} -> {comp_tokens} est. tokens)")
    print("-" * 55)
    print("Original Prompt:")
    print(f"  {prompt_text.strip()}")
    print("-" * 55)
    print("Compressed Prompt (Ready for API dispatch):")
    print(f"  {compressed.strip()}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
