import sys
import os
import pandas as pd
import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import RealLLMEnvironment, SimulatedEnvironment
from src.utils import calculate_reward, get_semantic_similarity

# --- Benchmark Dataset ---
BENCHMARK_DATA = [
    {"text": "Explain the concept of quantum entanglement.", "category": "QA"},
    {"text": "def quicksort(arr): return sorted(arr)", "category": "Code"},
    {"text": "How do you say 'Where is the library' in French?", "category": "Translation"},
    {"text": "Summarize the history of the Roman Empire in one paragraph.", "category": "Summarization"},
    {"text": "What's the weather like in New York today?", "category": "Chat"},
] * 300 # Expand to 125 samples

def run_benchmark(api_key=None):
    mode_names = ["Baseline", "Rule_Based", "LinUCB"]
    env = RealLLMEnvironment(api_key) if api_key else SimulatedEnvironment()
    
    all_logs = []
    
    for mode in mode_names:
        print(f"\n🚀 Running mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=5) if mode == "LinUCB" else None
        cumulative_reward = 0
        
        # Use tqdm for progress bar
        for i, data in enumerate(tqdm(BENCHMARK_DATA)):
            features = env.extract_features(data["text"])
            
            # Action Selection
            if mode == "Baseline": arm = 0
            elif mode == "Rule_Based": 
                arm = 0 if features[2] == 1.0 else (2 if len(data["text"]) > 100 else 1)
            elif mode == "LinUCB": 
                arm = agent.select_arm(features, step=i)
            
            # Execute
            res = env.execute_request(data["text"], arm)
            
            # Evaluation
            sem_score = get_semantic_similarity(res["answer"], res["answer"])
            reward, saving, lat_pen, fail_pen = calculate_reward(
                res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"], semantic_score=sem_score
            )
            
            # Update Agent
            if mode == "LinUCB":
                agent.update(arm, features, reward)
            
            cumulative_reward += reward
            all_logs.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "mode": mode,
                "category": data["category"],
                "step": i,
                "arm": arm,
                "reward": reward,
                "avg_reward": cumulative_reward / (i + 1),
                "saving_ratio": saving,
                "valid": res["valid"],
                "semantic_score": sem_score
            })

    # Save results
    df = pd.DataFrame(all_logs)
    filename = f"results/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Benchmark Complete! Results saved to: {filename}")
    
    # Generate quick summary
    summary = df.groupby("mode").agg({
        "reward": "mean",
        "saving_ratio": "mean",
        "valid": "mean",
        "semantic_score": "mean"
    })
    print("\n--- Summary Report ---")
    print(summary)

if __name__ == "__main__":
    # If API key is provided as argument, run Real mode, else Simulated
    key = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(key)
