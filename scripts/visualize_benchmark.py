import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Bar plot for Mean Reward and Saving Ratio by Mode
    summary = df.groupby('mode')[['reward', 'saving_ratio', 'valid']].mean().reset_index()
    print("\nMean Performance Summary:")
    print(summary)
    
    plt.figure(figsize=(15, 5))
    
    # Reward Plot
    plt.subplot(1, 3, 1)
    sns.barplot(data=summary, x='mode', y='reward', palette='viridis')
    plt.title('Mean Reward by Mode')
    plt.xticks(rotation=45)
    
    # Saving Ratio Plot
    plt.subplot(1, 3, 2)
    sns.barplot(data=summary, x='mode', y='saving_ratio', palette='magma')
    plt.title('Mean Saving Ratio by Mode')
    plt.xticks(rotation=45)
    
    # Validity Plot
    plt.subplot(1, 3, 3)
    sns.barplot(data=summary, x='mode', y='valid', palette='rocket')
    plt.title('Mean Validity Rate')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_img = 'results/benchmark_summary.png'
    plt.savefig(output_img)
    print(f"\nSummary chart saved to {output_img}")

    # 2. Cumulative Reward (Learning Curve) for LinUCB
    linucb_df = df[df['mode'] == 'LinUCB'].copy()
    if not linucb_df.empty:
        linucb_df['cumulative_reward'] = linucb_df['reward'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(linucb_df)), linucb_df['cumulative_reward'], label='LinUCB Cumulative Reward')
        plt.title('LinUCB Learning Curve (Cumulative Reward)')
        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/linucb_learning_curve.png')
        print("Learning curve saved to results/linucb_learning_curve.png")

if __name__ == "__main__":
    latest_file = "results/benchmark_20260530_194243.csv"
    if os.path.exists(latest_file):
        visualize_results(latest_file)
    else:
        print("File not found.")
