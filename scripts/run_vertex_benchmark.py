import sys
import os
import pandas as pd
import datetime
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import VertexAIEnvironment
from src.utils import calculate_reward, get_semantic_similarity

# --- Benchmark Dataset ---
BENCHMARK_DATA = [
    {"text": "Explain the concept of quantum entanglement.", "category": "QA"},
    {"text": "def quicksort(arr): return sorted(arr)", "category": "Code"},
    {"text": "How do you say 'Where is the library' in French?", "category": "Translation"},
    {"text": "Summarize the history of the Roman Empire in one paragraph.", "category": "Summarization"},
    {"text": "What's the weather like in New York today?", "category": "Chat"},
] * 20 # 100 samples for the Vertex test

def run_vertex_benchmark(project_id, key_path):
    print(f"🚀 Starting Large-Scale Vertex AI Benchmark")
    print(f"Project ID: {project_id}")
    
    # Initialize Vertex AI Environment
    env = VertexAIEnvironment(project_id=project_id, key_path=key_path)
    
    mode_names = ["Baseline", "LinUCB"] # Focused comparison
    all_logs = []
    
    for mode in mode_names:
        print(f"\n🔥 Running mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=5) if mode == "LinUCB" else None
        cumulative_reward = 0
        
        for i, data in enumerate(tqdm(BENCHMARK_DATA)):
            features = env.extract_features(data["text"])
            
            # Action Selection
            if mode == "Baseline": arm = 0
            elif mode == "LinUCB": 
                arm = agent.select_arm(features, step=i)
            
            # Execute Real Vertex Request (This consumes GCP credits)
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
            
            # Vertex AI has higher limits, but let's keep a small delay
            time.sleep(0.5)

    # Save results
    df = pd.DataFrame(all_logs)
    filename = f"results/vertex_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n✅ Vertex Benchmark Complete! Results saved to: {filename}")
    print("\n--- Summary Report ---")
    print(df.groupby("mode").agg({"reward": "mean", "saving_ratio": "mean", "valid": "mean"}))

if __name__ == "__main__":
    PROJECT_ID = "gen-lang-client-0752774059"
    KEY_PATH = "gcp-key.json"
    run_vertex_benchmark(PROJECT_ID, KEY_PATH)
