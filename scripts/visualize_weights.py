import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB

def plot_feature_importance(agent, feature_names):
    """
    Visualizes the learned theta weights for each compression arm.
    Positive values = Agent associates feature with HIGHER reward.
    Negative values = Agent associates feature with LOWER reward.
    """
    weights = agent.get_weights() # Shape: (n_arms, n_features)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(weights, annot=True, cmap="RdBu_r", center=0,
                xticklabels=feature_names, 
                yticklabels=[f"Arm {i} ({s})" for i, s in enumerate(['Cons.', 'Mod.', 'Aggr.'])])
    
    plt.title("LinUCB Learned Feature Importance (Theta Weights)")
    plt.xlabel("Linguistic Features")
    plt.ylabel("Compression Strategies")
    
    output_path = "assets/figure_4_weights.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Weight interpretability plot saved to: {output_path}")

if __name__ == "__main__":
    # Mock some data to demonstrate the plot
    agent = LinUCB(n_arms=3, n_features=5)
    
    # Feature 3 is 'Codeness'. Let's simulate that the agent learned 
    # that Arm 2 (Aggressive) is BAD for Code.
    mock_features = np.array([0.5, 0.5, 1.0, 0.5, 1.0]) # A code prompt
    agent.update(arm=2, features=mock_features, reward=-2.5) # Failed heavily
    
    feature_names = ["Length", "Diversity", "Codeness", "Entropy", "Bias"]
    plot_feature_importance(agent, feature_names)
