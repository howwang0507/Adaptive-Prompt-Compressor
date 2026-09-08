import sys
import os
import numpy as np
import optuna
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

# Large scale for robust Bayesian search
ITERATIONS = 5000
BENCHMARK_DATA = (REAL_DATA * (ITERATIONS // len(REAL_DATA) + 1))[:ITERATIONS]

def objective(trial):
    # Continuous Search Space
    l_save = trial.suggest_float("lambda_saving", 1.0, 5.0)
    l_fail = trial.suggest_float("lambda_failure", 5.0, 20.0)
    alpha = trial.suggest_float("alpha", 0.05, 2.0)
    
    env = SimulatedEnvironment()
    agent = LinUCB(n_arms=3, n_features=12, alpha=alpha)
    
    total_saving = 0
    total_valid = 0
    total_semantic = 0
    
    for i, data in enumerate(BENCHMARK_DATA):
        features = env.extract_features(data["text"])
        arm = agent.select_arm(features)
        
        res = env.execute_request(data["text"], arm)
        
        # Simulated Semantic Score
        if arm == 0:
            sem = 1.0
        elif arm == 1:
            sem = np.random.normal(0.96, 0.02)
        else:
            sem = np.random.normal(0.88, 0.05)
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
        
    avg_saving = total_saving / ITERATIONS
    avg_valid = total_valid / ITERATIONS
    avg_semantic = total_semantic / ITERATIONS
    
    # The Composite Score: We want to maximize the area of the (Saving, Valid, Semantic) cube
    composite_score = avg_saving * avg_valid * avg_semantic
    
    # Report metrics for logging
    trial.set_user_attr("avg_saving", avg_saving)
    trial.set_user_attr("avg_valid", avg_valid)
    trial.set_user_attr("avg_semantic", avg_semantic)
    
    return composite_score

def run_bayesian_optimization():
    print("🚀 Initiating Deep Bayesian Optimization (200 Trials)...")
    print("This will take a moment, but it will find the absolute limit.")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    
    print("\n🏆 OPTIMIZATION COMPLETE 🏆")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
        
    print("\nBest Performance Metrics:")
    print(f"  Composite Score: {study.best_value:.4f}")
    print(f"  Saving Ratio:    {study.best_trial.user_attrs['avg_saving']:.2%}")
    print(f"  Validity Rate:   {study.best_trial.user_attrs['avg_valid']:.2%}")
    print(f"  Semantic Score:  {study.best_trial.user_attrs['avg_semantic']:.4f}")
    
    # Save study results
    df = study.trials_dataframe()
    df.to_csv("results/deep_optimization_results.csv", index=False)
    print("Saved full history to results/deep_optimization_results.csv")

if __name__ == "__main__":
    run_bayesian_optimization()
