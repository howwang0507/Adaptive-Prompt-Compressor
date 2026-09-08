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

# Load data
DATA_PATH = "data/neurips_benchmark_full.json"
with open(DATA_PATH, 'r') as f:
    REAL_DATA = json.load(f)

# Medium scale for grid search speed
ITERATIONS = 2000
BENCHMARK_DATA = (REAL_DATA * (ITERATIONS // len(REAL_DATA) + 1))[:ITERATIONS]

def run_grid_search():
    # Grid search over lambda_saving and lambda_failure
    # We want to find the "Sweet Spot" where saving is high but validity doesn't tank
    savings_weights = [1.0, 2.0, 3.0]
    failure_penalties = [5.0, 10.0, 15.0]
    alphas = [0.1, 0.5, 1.0, 1.5]
    
    results = []
    env = SimulatedEnvironment()

    for l_save in savings_weights:
        for l_fail in failure_penalties:
            for alpha in alphas:
                print(f"Testing: SaveW={l_save}, FailW={l_fail}, Alpha={alpha}")
                agent = LinUCB(n_arms=3, n_features=12, alpha=alpha)
                
                total_saving = 0
                total_valid = 0
                total_semantic = 0
                
                for i, data in enumerate(BENCHMARK_DATA):
                    features = env.extract_features(data["text"])
                    arm = agent.select_arm(features)
                    
                    res = env.execute_request(data["text"], arm)
                    
                    # Simulated Semantic Score
                    if arm == 0: sem = 1.0
                    elif arm == 1: sem = np.random.normal(0.96, 0.02)
                    else: sem = np.random.normal(0.88, 0.05)
                    sem = max(min(sem, 1.0), 0.0)

                    reward, saving, _, _ = calculate_reward(
                        res["base_tokens"], res["comp_tokens"], res["latency"],
                        res["valid"], semantic_score=sem,
                        lambda_saving=l_save, lambda_failure=l_fail
                    )
                    
                    agent.update(arm, features, reward)
                    
                    total_saving += saving
                    total_valid += 1 if res["valid"] else 0
                    total_semantic += sem
                
                results.append({
                    "lambda_saving": l_save,
                    "lambda_failure": l_fail,
                    "alpha": alpha,
                    "avg_saving": total_saving / ITERATIONS,
                    "avg_valid": total_valid / ITERATIONS,
                    "avg_semantic": total_semantic / ITERATIONS,
                    "composite_score": (total_saving/ITERATIONS) * (total_valid/ITERATIONS) * (total_semantic/ITERATIONS)
                })

    df = pd.DataFrame(results)
    df.to_csv("results/grid_search_pareto.csv", index=False)
    
    best = df.loc[df['composite_score'].idxmax()]
    print("\n🏆 ABSOLUTE BEST CONFIGURATION FOUND:")
    print(best)
    return best

if __name__ == "__main__":
    run_grid_search()
