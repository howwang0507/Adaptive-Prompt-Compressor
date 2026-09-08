import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import SimulatedEnvironment
from src.utils import calculate_reward

# --- Load Real-World Benchmark Data ---
DATA_PATH = "data/neurips_benchmark_full.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'r') as f:
        REAL_DATA = json.load(f)
else:
    REAL_DATA = [{"text": "Sample", "category": "General"}]

ITERATIONS = 10000
BENCHMARK_DATA = (REAL_DATA * (ITERATIONS // len(REAL_DATA) + 1))[:ITERATIONS]

def run_safety_benchmark():
    # Comparing the previous optimized vs the new Safety-Constrained version
    mode_names = ["Baseline", "LinUCB_Optimized", "LinUCB_Safety"]
    env = SimulatedEnvironment()
    
    all_logs = []

    for mode in mode_names:
        print(f"\n🚀 Running: {mode}")
        alpha = 1.669 
        agent = LinUCB(n_arms=3, n_features=12, alpha=alpha) if "LinUCB" in mode else None
        cumulative_reward = 0

        for i, data in enumerate(tqdm(BENCHMARK_DATA)):
            features = env.extract_features(data["text"])

            # Action Selection
            if mode == "Baseline":
                arm = 0
            elif mode == "LinUCB_Optimized":
                arm = agent.select_arm(features, step=i)
            elif mode == "LinUCB_Safety":
                # --- SAFETY GATE ---
                # Manual score calculation to avoid AttributeError
                if data["category"] in ["Math", "Code"]:
                    p_safe = np.zeros(2) # Only Arm 0 and 1
                    x_transformed = agent.scaler.transform(np.array(features)).reshape(-1, 1)
                    for a in range(2):
                        theta = agent.A_inv[a] @ agent.b[a]
                        term1 = (theta.T @ x_transformed).item()
                        term2 = agent.alpha * np.sqrt((x_transformed.T @ agent.A_inv[a] @ x_transformed).item())
                        p_safe[a] = term1 + term2
                    arm = int(np.argmax(p_safe))
                else:
                    arm = agent.select_arm(features, step=i)

            # Execute
            res = env.execute_request(data["text"], arm)
            
            # Simulated Semantic Score logic
            # Baseline is perfect
            if arm == 0:
                sem_score = 1.0
            elif arm == 1:
                sem_score = max(min(np.random.normal(0.96, 0.02), 1.0), 0.0)
            else: # Arm 2 (Aggressive)
                sem_score = max(min(np.random.normal(0.85, 0.08), 1.0), 0.0)

            # Adjusting reward for Safety Mode: High failure penalty
            l_fail = 10.0 if mode == "LinUCB_Safety" else 2.5
            
            reward, saving, _, _ = calculate_reward(
                res["base_tokens"],
                res["comp_tokens"],
                res["latency"],
                res["valid"],
                semantic_score=sem_score,
                lambda_failure=l_fail
            )

            # Update Agent
            if agent:
                agent.update(arm, features, reward)

            cumulative_reward += reward
            
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

    # Save
    df = pd.DataFrame(all_logs)
    filename = "results/safety_neurips_benchmark.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Safety benchmark complete! Saved to {filename}")

if __name__ == "__main__":
    run_safety_benchmark()
