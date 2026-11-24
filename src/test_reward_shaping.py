
import gymnasium as gym

from reward_shaping import RewardShapingWrapper


env = gym.make("LunarLander-v3")
shaped_env = RewardShapingWrapper(env)

observation, info = shaped_env.reset(seed=42)

print("Testing Reward Shaping")
print("-" * 65)

for step in range(20):

    action = shaped_env.action_space.sample()

    observation, combined_reward, terminated, truncated, info = (
        shaped_env.step(action)
    )

    original_reward = info.get("original_reward", None)

    print(
        f"Step: {step + 1:02d} | "
        f"Combined Reward: {combined_reward:.4f} | "
        f"Original Reward: {original_reward}"
    )

    if terminated or truncated:
        break

shaped_env.close()

print("\nReward shaping test completed.")
