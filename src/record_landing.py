
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import os

# Create videos directory
os.makedirs("videos", exist_ok=True)

# Load the best-performing model
model = PPO.load("models/ppo_lunar_lander_shaped")

# Create environment with video recording
env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda episode_id: True,
    name_prefix="reward_shaped_landing"
)

for episode in range(5):
    observation, info = env.reset(seed=1000 + episode)

    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()

print("\nVideo recording completed!")
print("Check the videos/ folder.")
