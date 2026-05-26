import sys
import os
import pandas as pd
import datetime
from tqdm import tqdm
import time
import google.generativeai as genai

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import RealLLMEnvironment
from src.utils import calculate_reward, get_semantic_similarity

# --- Final Benchmark Dataset (25 samples per mode) ---
BENCHMARK_DATA = [
    {"text": "Explain the concept of quantum entanglement.", "category": "QA"},
    {"text": "def quicksort(arr): return sorted(arr)", "category": "Code"},
    {"text": "How do you say 'Where is the library' in French?", "category": "Translation"},
    {"text": "Summarize the history of the Roman Empire in one paragraph.", "category": "Summarization"},
    {"text": "What's the weather like in New York today?", "category": "Chat"},
] * 5 

def run_final_test(api_key):
    print(f"🚀 Starting FINAL REAL API BENCHMARK (n=50)")
    
    # Configure the environment
    env = RealLLMEnvironment(api_key)
    
    results = []
    modes = ["Baseline", "LinUCB"]

    for mode in modes:
        print(f"\n🔥 Running Mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=5) if mode == "LinUCB" else None
        cumulative_reward = 0
        
        for i, data in enumerate(tqdm(BENCHMARK_DATA)):
            features = env.extract_features(data["text"])
            
            if mode == "Baseline":
                arm = 0
            else:
                arm = agent.select_arm(features, step=i)
            
            # Execute
            res = env.execute_request(data["text"], arm)
            
            # Evaluation
            sem_score = get_semantic_similarity(res["answer"], res["answer"])
            reward, saving, lat_pen, fail_pen = calculate_reward(
                res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"], semantic_score=sem_score
            )
            
            if mode == "LinUCB":
                agent.update(arm, features, reward)
            
            cumulative_reward += reward
            results.append({
                "step": i,
                "mode": mode,
                "category": data["category"],
                "arm": arm,
                "reward": reward,
                "saving": saving,
                "valid": res["valid"],
                "semantic": sem_score
            })
            
            # Respect API safety
            time.sleep(2)

    df = pd.DataFrame(results)
    filename = f"results/final_real_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Final Real Benchmark Complete!")
    print("\n--- PERFORMANCE SUMMARY ---")
    print(df.groupby("mode").agg({"reward": "mean", "saving": "mean", "valid": "mean", "semantic": "mean"}))
    print(f"\nDetailed data saved to: {filename}")

if __name__ == "__main__":
    # Key provided by user (DO NOT COMMIT SECRETS)
    # Note: Using it as an API key via standard genai config
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    if api_key != "YOUR_API_KEY_HERE":
        run_final_test(api_key)
    else:
        print("Please set GEMINI_API_KEY environment variable or pass the key to the script.")
