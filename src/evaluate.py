
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment with graphics
env = gym.make("LunarLander-v3", render_mode="human")

# Load trained model
model = PPO.load("models/ppo_lunar_lander")

# Evaluate over 10 episodes
episodes = 10
rewards = []

for episode in range(episodes):

    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Predict the best action
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

    rewards.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}"
    )

print("\nEvaluation Results")
print("Average Reward:", sum(rewards) / episodes)
print("Best Reward:", max(rewards))
print("Worst Reward:", min(rewards))

env.close()
