"""
OpenAI Integration Demo - Adaptive Prompt Compressor
Demonstrates token reduction with OpenAI GPT-4o / GPT-4o-mini using LinUCB.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface import LinUCBCompressor


def main():
    print("🧠 Adaptive Prompt Compressor × OpenAI (GPT-4o) Demo\n")

    # Initialize compressor with OpenAI provider
    # Automatically picks up OPENAI_API_KEY from environment or falls back to simulation
    api_key = os.getenv("OPENAI_API_KEY")
    compressor = LinUCBCompressor(
        api_key=api_key,
        provider="openai",
        model_name="gpt-4o-mini",
        alpha=1.0,
    )

    print(f"Active Provider: {compressor.provider.upper()}")
    print("Target Model: gpt-4o-mini\n")

    # Sample prompt with verbose instructions (typical in RAG / system prompts)
    verbose_prompt = """
You are an expert full-stack software engineer. I need you to implement a robust, highly optimized, 
and clean Python function named `parse_server_logs` that takes a file path to an Apache or Nginx access log file, 
reads the contents line by line to conserve memory, extracts the IP addresses, HTTP status codes, and timestamps 
using regular expressions, counts the frequency of each unique status code (such as 200, 404, 500), 
and returns a structured Python dictionary containing the aggregated metrics. 
Please ensure that you handle FileNotFoundError gracefully and output only the code.
"""

    print("--- Original Prompt ---")
    print(f"Raw Length: {len(verbose_prompt.strip())} chars (approx {len(verbose_prompt.strip()) // 4} tokens)")

    # Execute LinUCB compression
    compressed_text, strategy, meta = compressor.compress(verbose_prompt)

    print("\n--- LinUCB Adaptive Compression ---")
    print(f"Selected Strategy : {strategy} (Arm {meta['arm']})")
    print(f"Compressed Length : {len(compressed_text)} chars (approx {len(compressed_text) // 4} tokens)")

    tokens_saved_pct = 100 - (len(compressed_text) / len(verbose_prompt.strip())) * 100
    print(f"Token Reduction   : {tokens_saved_pct:.1f}%\n")

    # Example 2: Chat / Documentation Summarization Prompt
    chat_prompt = """
Could you please provide a thorough, comprehensive, and exhaustive overview of how PostgreSQL 
handles multi-version concurrency control (MVCC)? Specifically, explain how dead tuples accumulate, 
how vacuuming works in the background, and what specific configuration knobs a database administrator 
can tune to prevent write amplification and transaction ID wraparound in high-throughput enterprise environments?
"""
    print("--- Example 2: Conversational / Long Documentation Prompt ---")
    print(f"Raw Length: {len(chat_prompt.strip())} chars (approx {len(chat_prompt.strip()) // 4} tokens)")

    compressed_chat, chat_strat, chat_meta = compressor.compress(chat_prompt)
    print(f"Selected Strategy : {chat_strat} (Arm {chat_meta['arm']})")
    print(f"Compressed Length : {len(compressed_chat)} chars (approx {len(compressed_chat) // 4} tokens)")
    chat_savings = 100 - (len(compressed_chat) / len(chat_prompt.strip())) * 100
    print(f"Token Reduction   : {chat_savings:.1f}%")
    print(f"Compressed Output :\n'{compressed_chat.strip()}'\n")


if __name__ == "__main__":
    main()
