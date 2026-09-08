import sys
import os
import pandas as pd
from tqdm import tqdm
import json
import time
import numpy as np
from google import genai

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import LinUCB
from src.environment import BaseLLMEnvironment
from src.utils import calculate_reward

DATA_PATH = "data/neurips_benchmark_full.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'r') as f:
        REAL_DATA = json.load(f)
else:
    print("Error: data/neurips_benchmark_full.json not found.")
    sys.exit(1)

SAMPLE_SIZE = 50 
np.random.seed(42)
validation_indices = np.random.choice(len(REAL_DATA), min(SAMPLE_SIZE, len(REAL_DATA)), replace=False)
VALIDATION_DATA = [REAL_DATA[i] for i in validation_indices]

class ModernGenAIEnv(BaseLLMEnvironment):
    def __init__(self, api_key):
        super().__init__()
        # Initialize the modern, officially supported google-genai SDK
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash" # Use a modern model if possible, fallback to 1.5 if needed
        # Fallback logic if 2.5 is not accessible
        try:
             self.client.models.get(model="gemini-2.5-flash")
        except Exception:
             self.model_name = "gemini-1.5-flash"

    def execute_request(self, text, arm):
        base_tokens = len(text) // 4
        compressed_text = self.compress_prompt(text, arm)
        if not compressed_text or len(compressed_text.strip()) == 0:
            compressed_text = text # Fallback to original if compression stripped everything
        comp_tokens = len(compressed_text) // 4

        start_time = time.time()
        answer, is_valid = "", True
        
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=compressed_text,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=100
                    )
                )
                answer = response.text
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(25 * (attempt + 1)) # Wait longer for free tier
                    if attempt == 2:
                        answer, is_valid = f"API Error (Rate Limit): {str(e)}", False
                else:
                    answer, is_valid = f"API Error: {str(e)}", False
                    break

        return {
            "base_tokens": base_tokens,
            "comp_tokens": comp_tokens,
            "latency": (time.time() - start_time) * 1000,
            "valid": is_valid,
            "answer": answer,
        }

def run_real_api_validation(api_key):
    print("🚀 Initiating Validation using modern `google-genai` SDK...")
    
    env = ModernGenAIEnv(api_key)
    
    modes = ["Baseline", "LinUCB_Optimum"]
    all_logs = []

    for mode in modes:
        print(f"Running mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=12, alpha=1.13) if "LinUCB" in mode else None
        
        for i, data in enumerate(tqdm(VALIDATION_DATA)):
            features = env.extract_features(data["text"])
            
            arm = 0 if mode == "Baseline" else agent.select_arm(features)
            res = env.execute_request(data["text"], arm)
            
            sem_score = 1.0 if res["valid"] else 0.0 
            
            reward, saving, _, _ = calculate_reward(
                res["base_tokens"], res["comp_tokens"], res["latency"],
                res["valid"], semantic_score=sem_score,
                lambda_saving=4.92, lambda_failure=5.01
            )
            
            if agent:
                agent.update(arm, features, reward)
                
            all_logs.append({
                "mode": mode,
                "saving_ratio": saving,
                "valid": res["valid"],
                "latency_ms": res["latency"],
                "error": res["answer"] if not res["valid"] else ""
            })
            
            if not res["valid"] and "Error:" in res["answer"] and "429" not in res["answer"] and "Rate Limit" not in res["answer"]:
                print(f"\n❌ Immediate failure detected: {res['answer']}")
                return

    df = pd.DataFrame(all_logs)
    df.to_csv("results/real_api_validation.csv", index=False)
    
    print("\n✅ Real API Validation Complete!")
    print("\n--- Summary (Real LLM Responses) ---")
    summary = df.groupby('mode')[['saving_ratio', 'valid', 'latency_ms']].mean().reset_index()
    print(summary)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="API Key")
    args = parser.parse_args()
    
    run_real_api_validation(args.key)
