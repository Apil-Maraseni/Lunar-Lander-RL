
import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the best-performing model
model = PPO.load("models/ppo_lunar_lander_shaped")

# Create a graphical environment
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

episode = 1
total_reward = 0

while episode <= 5:
    action, _ = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode {episode} | Reward: {total_reward:.2f}")

        episode += 1
        if episode <= 5:
            time.sleep(2)
            observation, info = env.reset()
            total_reward = 0

env.close()
