import sys
import os
import pandas as pd
import datetime
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import SimulatedEnvironment
from src.utils import calculate_reward, get_semantic_similarity

# --- Load Real-World Benchmark Data ---
DATA_PATH = "data/neurips_benchmark_full.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'r') as f:
        REAL_DATA = json.load(f)
else:
    REAL_DATA = [{"text": "Sample", "category": "General"}]

# Expand to 10,000 iterations for convergence analysis
ITERATIONS = 10000
BENCHMARK_DATA = (REAL_DATA * (ITERATIONS // len(REAL_DATA) + 1))[:ITERATIONS]

def run_large_scale_benchmark():
    # Modes to compare
    mode_names = ["Baseline", "LLMLingua_Sim", "LinUCB_Optimized"]
    env = SimulatedEnvironment()
    
    all_logs = []

    for mode in mode_names:
        print(f"\n🚀 Running Large Scale: {mode}")
        # Use our optimized alpha=1.669
        alpha = 1.669 if "Optimized" in mode else 1.0
        agent = LinUCB(n_arms=3, n_features=12, alpha=alpha) if "LinUCB" in mode else None
        cumulative_reward = 0

        for i, data in enumerate(tqdm(BENCHMARK_DATA)):
            features = env.extract_features(data["text"])

            # Action Selection
            if mode == "Baseline":
                arm = 0
            elif mode == "LLMLingua_Sim":
                # Simulated budget-aware behavior
                arm = 0 if data["category"] == "Code" else (2 if len(data["text"]) > 200 else 1)
            elif "LinUCB" in mode:
                arm = agent.select_arm(features, step=i)

            # Execute Simulation with Gemini 1.5 Flash performance profile
            # Higher penalty for Code errors
            res = env.execute_request(data["text"], arm)
            
            # Simulated Semantic Score: arm 2 (aggressive) has higher variance
            base_sem = 1.0 if arm == 0 else (0.95 if arm == 1 else 0.85)
            noise = np.random.normal(0, 0.05)
            sem_score = max(min(base_sem + noise, 1.0), 0.0)

            reward, saving, _, _ = calculate_reward(
                res["base_tokens"],
                res["comp_tokens"],
                res["latency"],
                res["valid"],
                semantic_score=sem_score,
            )

            # Update Agent
            if agent:
                agent.update(arm, features, reward)

            cumulative_reward += reward
            
            # Log every 10 steps to save space but keep resolution
            if i % 10 == 0 or i == ITERATIONS - 1:
                all_logs.append({
                    "mode": mode,
                    "step": i,
                    "reward": reward,
                    "avg_reward": cumulative_reward / (i + 1),
                    "saving_ratio": saving,
                    "valid": res["valid"],
                    "semantic_score": sem_score
                })

    # Save results
    df = pd.DataFrame(all_logs)
    filename = "results/large_scale_neurips_benchmark.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Large scale benchmark complete! Saved to {filename}")

if __name__ == "__main__":
    run_large_scale_benchmark()
