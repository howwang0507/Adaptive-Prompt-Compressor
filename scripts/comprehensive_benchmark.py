import sys
import os
import pandas as pd
import datetime
import json
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import RealLLMEnvironment
from src.utils import calculate_reward, get_semantic_similarity
from src.database import ExperimentDB

def run_comprehensive_benchmark(api_key, num_repeats=5):
    print("🚀 Initializing Comprehensive API Benchmark (Real Gemini API)")
    
    # 1. Setup Environment and Database
    env = RealLLMEnvironment(api_key)
    db = ExperimentDB()
    
    # 2. Load Prompts
    with open("data/benchmark_prompts.json", "r") as f:
        base_prompts = json.load(f)
    
    # Expand dataset to ensure learning can be observed
    test_dataset = base_prompts * num_repeats
    
    results = []
    modes = ["Baseline", "LinUCB"]

    for mode in modes:
        print(f"\n🔥 Mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=5, alpha=1.0) if mode == "LinUCB" else None
        
        for i, data in enumerate(tqdm(test_dataset)):
            features = env.extract_features(data["text"])
            
            # Action Selection
            if mode == "Baseline":
                arm = 0
            else:
                arm = agent.select_arm(features, step=i)
            
            # Execute REAL API Request
            try:
                res = env.execute_request(data["text"], arm)
                
                # Evaluation
                sem_score = get_semantic_similarity(data["text"], res["answer"])
                reward, saving, lat_pen, fail_pen = calculate_reward(
                    res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"], semantic_score=sem_score
                )
                
                # Update Agent if learning
                if mode == "LinUCB":
                    agent.update(arm, features, reward)
                
                # Record Entry
                entry = {
                    "mode": mode,
                    "category": data["category"],
                    "arm": arm,
                    "reward": reward,
                    "token_saved": saving,
                    "success_rate": 1 if res["valid"] else 0,
                    "semantic_score": sem_score,
                    "latency": res["latency"]
                }
                
                # Save to SQL DB for professional analysis
                db.log_trial(entry)
                results.append(entry)
                
                # Print periodic insight
                if i % 5 == 0:
                    print(f" [Step {i}] Rew: {reward:.2f} | Sav: {saving:.1%} | Valid: {res['valid']}")
                
                # Respect API Quota (Gemini Free Tier has strict RPM)
                time.sleep(4) 
                
            except Exception as e:
                print(f"⚠️ API Critical Failure: {e}")
                time.sleep(10) # Cooling down

    # 3. Final Report Generation
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("✨ COMPREHENSIVE BENCHMARK COMPLETE ✨")
    print("="*40)
    
    summary = df.groupby(["mode"]).agg({
        "reward": "mean",
        "token_saved": "mean",
        "success_rate": "mean",
        "semantic_score": "mean",
        "latency": "mean"
    })
    print("\n--- PERFORMANCE OVERVIEW ---")
    print(summary)
    
    # Category Analysis for LinUCB
    if "LinUCB" in df["mode"].values:
        print("\n--- LINUCB TASK SPECIALIZATION ---")
        cat_analysis = df[df["mode"]=="LinUCB"].groupby("category")["arm"].value_counts(normalize=True).unstack(fill_value=0)
        print(cat_analysis)

    # Save to CSV as backup
    report_path = f"results/final_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    print(f"\n📊 Detailed CSV report saved to: {report_path}")
    print("🗄️ SQL Database updated: results/experiments.db")

if __name__ == "__main__":
    import os
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("❌ Error: GEMINI_API_KEY is not set.")
        print("Usage: export GEMINI_API_KEY='your_key' && python scripts/comprehensive_benchmark.py")
    else:
        run_comprehensive_benchmark(key)
