
import gymnasium as gym
from stable_baselines3 import PPO

# Load the reward-shaped model
model = PPO.load("models/ppo_lunar_lander_shaped")

# Use the original environment (no reward-shaping wrapper)
env = gym.make("LunarLander-v3")

rewards = []

for episode in range(10):
    observation, info = env.reset(seed=1000 + episode)
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()

print("\nEvaluation Results")
print("-------------------------")
print(f"Average Reward: {sum(rewards) / len(rewards):.2f}")
print(f"Best Reward: {max(rewards):.2f}")
print(f"Worst Reward: {min(rewards):.2f}")
print(f"Successful Landings: {sum(r > 0 for r in rewards)}/10")
