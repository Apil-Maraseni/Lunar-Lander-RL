
import gymnasium as gym

env = gym.make("LunarLander-v3")

print("Observation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

observation, info = env.reset()

print("\nInitial Observation:")
for i, value in enumerate(observation):
    print(f"Feature {i}: {value}")

print("\nAvailable Actions:")
print(env.action_space.n)

env.close()
