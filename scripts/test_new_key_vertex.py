import sys
import os
import pandas as pd
from tqdm import tqdm
import json
import time
import numpy as np
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

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

class CustomVertexEnv(BaseLLMEnvironment):
    def __init__(self, api_key):
        super().__init__()
        # Initialize Vertex AI with the specific api_key if possible, 
        # but usually Vertex expects a service account JSON.
        # If this is an API key for the new Agent Platform SDK, we might need a different init.
        # Let's try the standard client options if it's a raw key, or fallback to default if it's meant for GenAI.
        
        # NOTE: Vertex AI primarily uses OAuth (Service Accounts), not raw API keys. 
        # If the user provided an "AIza..." key, it's almost certainly for the Developer API (generativelanguage.googleapis.com)
        # However, to humor the request, we will attempt to set it in the environment if there's a specialized SDK.
        
        # For now, we assume standard Vertex initialization and rely on the environment being correctly configured 
        # or we inform the user of the distinction.
        print("Initializing custom Vertex Env. Note: Vertex AI usually requires a service account JSON, not an API key.")
        
        self.model = GenerativeModel("gemini-1.5-flash")
        self.safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def execute_request(self, text, arm):
        base_tokens = len(text) // 4
        compressed_text = self.compress_prompt(text, arm)
        comp_tokens = len(compressed_text) // 4 if compressed_text else 0

        start_time = time.time()
        answer, is_valid = "", True
        
        try:
            response = self.model.generate_content(
                compressed_text,
                safety_settings=self.safety_config,
                generation_config={"temperature": 0.2, "max_output_tokens": 100},
            )
            answer = response.text
        except Exception as e:
            answer, is_valid = f"Vertex Error: {str(e)}", False

        return {
            "base_tokens": base_tokens,
            "comp_tokens": comp_tokens,
            "latency": (time.time() - start_time) * 1000,
            "valid": is_valid,
            "answer": answer,
        }

def run_vertex_validation(api_key):
    # Set the key in the environment just in case it's picked up by the underlying library
    os.environ["GEMINI_API_KEY"] = api_key
    print(f"🚀 Initiating Validation using Vertex SDK structure...")
    
    env = CustomVertexEnv(api_key)
    
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
                
            all_logs.append({"mode": mode, "valid": res["valid"], "error": res["answer"] if not res["valid"] else ""})
            time.sleep(1.0)
            
            # Break early if it's just a credential error
            if not res["valid"] and "Error:" in res["answer"]:
                print(f"\n❌ Immediate failure detected: {res['answer']}")
                return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="API Key")
    args = parser.parse_args()
    
    run_vertex_validation(args.key)
