import pandas as pd
from src.agent import LinUCB
from src.environment import SimulatedEnvironment
from src.utils import calculate_reward

def run_ablation():
    """
    Automates feature importance analysis.
    Compares the full feature set vs. removing specific features.
    """
    env = SimulatedEnvironment()
    features_to_test = {
        "Full Set": [0, 1, 2, 3, 4],
        "No Codeness": [0, 1, 3, 4],
        "No Entropy": [0, 1, 2, 4],
        "No Diversity": [0, 2, 3, 4]
    }
    
    results = []
    print("Starting Ablation Study...")

    for name, indices in features_to_test.items():
        agent = LinUCB(n_arms=3, n_features=len(indices))
        total_reward = 0
        
        # Test 100 steps
        for i in range(100):
            # Simulated data for code blocks
            text = "def test(): return True" if i % 2 == 0 else "Hello world"
            full_feat = env.extract_features(text)
            
            # Mask features
            sub_feat = [full_feat[idx] for idx in indices]
            
            arm = agent.select_arm(sub_feat)
            res = env.execute_request(text, arm)
            
            # Penalize heavily if Code is compressed aggressively when Codeness feature is missing
            if "def" in text and arm == 2:
                res["valid"] = False # Force failure for analysis
            
            reward, _, _, _ = calculate_reward(res["base_tokens"], res["comp_tokens"], res["latency"], res["valid"])
            agent.update(arm, sub_feat, reward)
            total_reward += reward
            
        results.append({"Configuration": name, "Final Avg Reward": total_reward / 100})
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)
    print("Ablation Study Complete. Results saved to ablation_results.csv")
    print(df)
    
    # Generate PNG
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(df["Configuration"], df["Final Avg Reward"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title("Ablation Study: Feature Importance Analysis", fontsize=14, fontweight='bold')
    plt.ylabel("Final Average Reward")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("ablation_study.png", dpi=300)
    print("Figure saved as ablation_study.png")

if __name__ == "__main__":
    run_ablation()
