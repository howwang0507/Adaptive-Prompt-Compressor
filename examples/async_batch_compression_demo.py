"""
High-Throughput Asynchronous Batch Prompt Compression Cookbook
==============================================================
Demonstrates how AsyncLinUCBCompressor processes high-volume prompt streams
concurrently via asyncio.gather, reducing end-to-end token latency and cloud API costs.
"""

import asyncio
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.async_interface import AsyncLinUCBCompressor


async def run_batch_compression_demo():
    print("=" * 70)
    print("⚡ High-Throughput Async Batch Prompt Compression Demo")
    print("=" * 70)

    # Initialize async compressor with offline simulation
    async_compressor = AsyncLinUCBCompressor()

    prompts = [
        "Please carefully analyze and summarize the quarterly balance sheet of Acme Corp.",
        "Write a Python script that calculates rolling volatility for SPY ETF daily prices.",
        "Could you explain the technical difference between optimistic and pessimistic concurrency control?",
        "Review this pull request and check whether all database connections are properly pooled.",
        "Implement a binary search tree traversal algorithm in Rust with proper lifetime annotations.",
    ]

    print(f"\nSubmitting {len(prompts)} prompts concurrently to AsyncLinUCBCompressor...")
    t0 = time.perf_counter()

    # Run parallel batch compression
    results = await asyncio.gather(*[async_compressor.compress_async(p) for p in prompts])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"✓ All {len(prompts)} prompts processed concurrently in {elapsed_ms:.2f} ms!")
    print("\n[Batch Execution Breakdown]")
    print("-" * 70)
    for idx, (compressed_text, strategy, meta) in enumerate(results, 1):
        orig_len = len(prompts[idx - 1].split())
        comp_len = len(compressed_text.split())
        saved_pct = (1.0 - comp_len / max(1, orig_len)) * 100.0
        print(f"[{idx}] Strategy: {strategy:<12} | Arm: {meta['arm']} | Savings: {saved_pct:5.1f}%")
        print(f"    Original:   {prompts[idx - 1]}")
        print(f"    Compressed: {compressed_text}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(run_batch_compression_demo())
