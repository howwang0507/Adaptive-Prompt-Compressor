import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_large_scale(file_path):
    df = pd.read_csv(file_path)
    
    plt.figure(figsize=(15, 6))
    
    # 1. Learning Curve (Average Reward Over Time)
    plt.subplot(1, 2, 1)
    for mode in df['mode'].unique():
        subset = df[df['mode'] == mode]
        plt.plot(subset['step'], subset['avg_reward'], label=mode)
    
    plt.title('Learning Curve: Average Reward vs. Steps (N=10,000)')
    plt.xlabel('Training Steps')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Saving Ratio Distribution
    plt.subplot(1, 2, 2)
    # Get final summary stats
    summary = df.groupby('mode')[['saving_ratio', 'valid', 'semantic_score']].mean().reset_index()
    print("\n--- Final Large Scale Summary ---")
    print(summary)
    
    sns.barplot(data=summary, x='mode', y='saving_ratio', palette='viridis')
    plt.title('Mean Token Saving Ratio (Real-world Datasets)')
    plt.ylabel('Saving Ratio (0.0 - 1.0)')
    
    plt.tight_layout()
    plt.savefig('results/neurips_large_scale_analysis.png')
    print("\n✅ NeurIPS Analysis Plot saved to results/neurips_large_scale_analysis.png")

if __name__ == "__main__":
    plot_large_scale("results/safety_neurips_benchmark.csv")
