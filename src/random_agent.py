
import gymnasium as gym
import time

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Start an episode
observation, info = env.reset()

print("Initial observation:")
print(observation)

for step in range(1000):

    # Select a random action
    action = env.action_space.sample()

    # Perform the action
    observation, reward, terminated, truncated, info = env.step(action)

    print("Step:", step)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    # Check whether the episode ended
    if terminated or truncated:
        print("Episode finished!")
        break

env.close()
