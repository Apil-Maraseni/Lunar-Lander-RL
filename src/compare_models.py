
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

models = {
    "Baseline PPO": "models/ppo_lunar_lander",
    "Tuned PPO": "models/ppo_lunar_lander_tuned",
    "Reward-Shaped PPO": "models/ppo_lunar_lander_shaped"
}

def evaluate_model(model_path):
    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")

    rewards = []

    for episode in range(100):
        observation, info = env.reset(seed=1000 + episode)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    return {
        "average": np.mean(rewards),
        "std": np.std(rewards),
        "best": np.max(rewards),
        "worst": np.min(rewards),
        "positive": np.sum(np.array(rewards) > 0)
    }


results = {}

for name, path in models.items():
    print(f"\nEvaluating {name} (100 episodes)...")
    results[name] = evaluate_model(path)

print("\n========== 100-EPISODE COMPARISON ==========")

for name, result in results.items():
    print(f"\n{name}")
    print("-" * 40)
    print(f"Average Reward: {result['average']:.2f}")
    print(f"Std Deviation: {result['std']:.2f}")
    print(f"Best Reward: {result['best']:.2f}")
    print(f"Worst Reward: {result['worst']:.2f}")
    print(f"Positive Episodes: {result['positive']}/100")
