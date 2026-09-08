"""
1-Line Drop-In OpenAI SDK Wrapper Example
Demonstrates how to use `wrap_openai_client` to transparently compress prompts
before calling OpenAI's chat completions API.
"""

import os
import sys
from openai import OpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrations.openai_client import wrap_openai_client


def main():
    print("🔌 OpenAI SDK 1-Line Drop-in Wrapper Demo\n")

    # Standard OpenAI client initialization
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set. Using local simulation environment for demo.")
        from unittest.mock import MagicMock
        raw_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
        raw_client.chat.completions.create.return_value = mock_response
    else:
        raw_client = OpenAI(api_key=api_key)

    # Wrap client in one line:
    client = wrap_openai_client(raw_client)

    prompt = """
Could you please write a clear, robust Python function to calculate the Fibonacci series up to n elements?
Please include comments and handle edge cases where n <= 0.
"""

    print("Sending prompt through wrapped client...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise code assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    print("\n--- Execution Finished ---")
    print(f"Response: {response.choices[0].message.content.strip()}")
    print("\n--- Automatic Compression Telemetry ---")
    print(f"Compression Metadata: {response.compression_meta}")


if __name__ == "__main__":
    main()
