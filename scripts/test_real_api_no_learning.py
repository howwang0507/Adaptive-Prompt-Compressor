import sys
import os
import pandas as pd
import datetime
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import RealLLMEnvironment
from src.utils import calculate_reward, get_semantic_similarity

# --- Test Dataset (10 samples for safety) ---
TEST_DATA = [
    {"text": "Explain the concept of quantum entanglement.", "category": "QA"},
    {"text": "def quicksort(arr): return sorted(arr)", "category": "Code"},
    {
        "text": "How do you say 'Where is the library' in French?",
        "category": "Translation",
    },
    {
        "text": "Summarize the history of the Roman Empire in one paragraph.",
        "category": "Summarization",
    },
    {"text": "What's the weather like in New York today?", "category": "Chat"},
] * 2


def run_real_test(api_key):
    print("🚀 Starting Real API Test (Mode: NO LEARNING)")
    env = RealLLMEnvironment(api_key)
    # Initialize a fresh agent
    agent = LinUCB(n_arms=3, n_features=5)

    results = []

    for i, data in enumerate(tqdm(TEST_DATA)):
        features = env.extract_features(data["text"])

        # Select arm using initial (untrained) weights
        arm = agent.select_arm(features, step=i)

        # Execute Real Request
        print(f"\n[Step {i}] Category: {data['category']} | Selected Arm: {arm}")
        res = env.execute_request(data["text"], arm)

        if res["valid"]:
            print(f"✅ Success! Response preview: {res['answer'][:50]}...")
        else:
            print(f"❌ Failed. Error: {res['answer']}")

        # Evaluation
        sem_score = get_semantic_similarity(res["answer"], res["answer"])
        reward, saving, lat_pen, fail_pen = calculate_reward(
            res["base_tokens"],
            res["comp_tokens"],
            res["latency"],
            res["valid"],
            semantic_score=sem_score,
        )

        # --- LEARNING DISABLED ---
        # agent.update(arm, features, reward)

        results.append(
            {
                "step": i,
                "category": data["category"],
                "arm": arm,
                "reward": reward,
                "saving": saving,
                "valid": res["valid"],
            }
        )

        # Small delay between requests to be safe
        time.sleep(2)

    df = pd.DataFrame(results)
    print("\n--- Real API Test Summary ---")
    print(
        df.groupby("category").agg(
            {"reward": "mean", "saving": "mean", "valid": "mean"}
        )
    )

    filename = (
        f"results/real_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(filename, index=False)
    print(f"\nData saved to: {filename}")


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    run_real_test(api_key)
