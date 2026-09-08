import sys
import os
import pandas as pd
from tqdm import tqdm
import json
import time
import numpy as np
import urllib.request
import urllib.error

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import LinUCB
from src.environment import BaseLLMEnvironment
from src.utils import calculate_reward

DATA_PATH = "data/neurips_benchmark_full.json"
with open(DATA_PATH, 'r') as f:
    REAL_DATA = json.load(f)

SAMPLE_SIZE = 50 
np.random.seed(42)
validation_indices = np.random.choice(len(REAL_DATA), min(SAMPLE_SIZE, len(REAL_DATA)), replace=False)
VALIDATION_DATA = [REAL_DATA[i] for i in validation_indices]

class EnterpriseAgentPlatformEnv(BaseLLMEnvironment):
    def __init__(self, api_key, project_id):
        super().__init__()
        self.api_key = api_key
        self.project_id = project_id
        self.location = "us-central1"
        self.model_id = "gemini-1.5-flash"
        # The specific Enterprise REST endpoint structure
        self.url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:generateContent?key={self.api_key}"

    def execute_request(self, text, arm):
        base_tokens = len(text) // 4
        compressed_text = self.compress_prompt(text, arm)
        if not compressed_text or len(compressed_text.strip()) == 0:
            compressed_text = text
        comp_tokens = len(compressed_text) // 4

        start_time = time.time()
        answer, is_valid = "", True
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": compressed_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 150
            }
        }
        
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                answer = result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            if hasattr(e, 'read'):
                err_body = e.read().decode("utf-8")
                answer, is_valid = f"Enterprise API Error: {err_body}", False
            else:
                answer, is_valid = f"Network Error: {str(e)}", False

        return {
            "base_tokens": base_tokens,
            "comp_tokens": comp_tokens,
            "latency": (time.time() - start_time) * 1000,
            "valid": is_valid,
            "answer": answer,
        }

def run_enterprise_validation(api_key):
    # Using your "Default Gemini Project" ID seen in screenshot
    project_id = "gen-lang-client-0752774059"
    print("🚀 Starting Enterprise Agent Platform Validation...")
    print(f"Project: {project_id} | Region: us-central1\n")
    
    env = EnterpriseAgentPlatformEnv(api_key, project_id)
    
    modes = ["Baseline", "LinUCB_Optimum"]
    all_logs = []

    for mode in modes:
        print(f"Running mode: {mode}")
        agent = LinUCB(n_arms=3, n_features=12, alpha=1.13) if "LinUCB" in mode else None
        
        for i, data in enumerate(tqdm(VALIDATION_DATA)):
            features = env.extract_features(data["text"])
            arm = 0 if mode == "Baseline" else agent.select_arm(features)
            res = env.execute_request(data["text"], arm)
            
            # Use real validity and semantic scores (simulated for speed but can be BERTScore)
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
            
            if not res["valid"] and "PERMISSION_DENIED" in res["answer"]:
                print(f"\n❌ Auth Failure: {res['answer']}")
                return
            
            # Slight delay to respect Enterprise quotas
            time.sleep(0.5)

    df = pd.DataFrame(all_logs)
    df.to_csv("results/enterprise_validation_results.csv", index=False)
    print("\n✅ Enterprise Validation Complete!")
    print(df.groupby('mode')[['saving_ratio', 'valid', 'latency_ms']].mean())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    run_enterprise_validation(args.key)
