
import gymnasium as gym
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

    for episode in range(10):
        observation, info = env.reset(seed=1000 + episode)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    return {
        "average": sum(rewards) / len(rewards),
        "best": max(rewards),
        "worst": min(rewards),
        "positive": sum(r > 0 for r in rewards)
    }


results = {}

for name, path in models.items():
    print(f"\nEvaluating {name}...")
    results[name] = evaluate_model(path)

print("\n========== FINAL MODEL COMPARISON ==========")

for name, result in results.items():
    print(f"\n{name}")
    print("-" * 35)
    print(f"Average Reward: {result['average']:.2f}")
    print(f"Best Reward: {result['best']:.2f}")
    print(f"Worst Reward: {result['worst']:.2f}")
    print(f"Positive-Reward Episodes: {result['positive']}/10")
