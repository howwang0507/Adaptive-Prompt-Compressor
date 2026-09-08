import pandas as pd
from datasets import load_dataset
import os
import json

def download_and_save():
    print("📥 Downloading GSM8K (Math Reasoning)...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        gsm8k_df = pd.DataFrame(gsm8k)
        gsm8k_df['category'] = 'Math'
        gsm8k_samples = gsm8k_df[['question', 'category']].rename(columns={'question': 'text'}).sample(250)
    except Exception as e:
        print(f"Failed to download GSM8K: {e}")
        gsm8k_samples = pd.DataFrame()

    print("📥 Downloading HumanEval (Code Generation)...")
    try:
        # Note: HumanEval might require specific loading or a proxy dataset if not available
        he = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        he_df = pd.DataFrame(he)
        he_df['category'] = 'Code'
        he_samples = he_df[['prompt', 'category']].rename(columns={'prompt': 'text'}).sample(250)
    except Exception as e:
        print(f"Failed to download HumanEval: {e}")
        he_samples = pd.DataFrame()

    # Combine
    final_df = pd.concat([gsm8k_samples, he_samples], ignore_index=True)
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/neurips_benchmark_full.json"
    final_df.to_json(output_path, orient='records', indent=4)
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"Total samples: {len(final_df)}")

if __name__ == "__main__":
    download_and_save()
