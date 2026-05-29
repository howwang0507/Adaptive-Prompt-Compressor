import optuna
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LinUCB
from src.environment import SimulatedEnvironment


def objective(trial):
    # Hyperparameters to optimize
    alpha = trial.suggest_float("alpha", 0.1, 2.0)

    env = SimulatedEnvironment()
    agent = LinUCB(n_arms=3, n_features=5, alpha=alpha)

    total_reward = 0
    steps = 500

    # Run a mini-benchmark for this trial
    for step in range(steps):
        # Sample a random task for tuning
        prompt = "Sample task text for tuning"
        features = env.extract_features(prompt)
        arm = agent.select_arm(features)

        # Simulate feedback
        res = env.execute_request(prompt, arm)
        reward, _, _, _ = env.get_reward_logic(res)  # Abstracted reward logic

        agent.update(arm, features, reward)
        total_reward += reward

    return total_reward / steps


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\n🏆 Optimization Complete!")
    print(f"Best Alpha: {study.best_params['alpha']}")
    print(f"Best Mean Reward: {study.best_value}")
