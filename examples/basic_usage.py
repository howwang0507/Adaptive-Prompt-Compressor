import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface import LinUCBCompressor

def main():
    print("🧠 Adaptive Prompt Compressor - Basic Usage Example\n")
    
    # Initialize the compressor
    # If no API key is provided, it defaults to SimulatedEnvironment for local testing
    api_key = os.getenv("GEMINI_API_KEY")
    compressor = LinUCBCompressor(api_key=api_key, model_name="gemini-1.5-flash")

    # Example 1: Technical Prompt (Code)
    # The agent should automatically recognize this as technical syntax and preserve it (Arm 0)
    code_prompt = """
def calculate_fibonacci(n):
    # Calculate the nth fibonacci number
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
    print("--- Example 1: Code Prompt ---")
    compressed_code, strategy, _ = compressor.compress(code_prompt)
    print(f"Original Length: {len(code_prompt)} chars")
    print(f"Selected Strategy: {strategy}")
    print(f"Tokens Saved: {100 - (len(compressed_code)/len(code_prompt))*100:.1f}%\n")

    # Example 2: Conversational/Summarization Prompt
    # The agent should apply more aggressive POS-aware compression (Arm 2)
    chat_prompt = """
Could you please write a very detailed and robust Python function that counts the frequency of each word in a given text file? Please make sure to handle any potential FileNotFoundError exceptions that might occur if the file is missing, and also ignore common English stopwords like 'the', 'a', and 'an'.
"""
    print("--- Example 2: Chat/Task Prompt ---")
    compressed_chat, strategy, _ = compressor.compress(chat_prompt)
    print(f"Original Length: {len(chat_prompt)} chars")
    print(f"Selected Strategy: {strategy}")
    print(f"Compressed Result:\n'{compressed_chat}'")
    print(f"Tokens Saved: {100 - (len(compressed_chat)/len(chat_prompt))*100:.1f}%\n")

if __name__ == "__main__":
    main()
