
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_shaping import RewardShapingWrapper


# Create the LunarLander environment
env = gym.make("LunarLander-v3")

# Apply our custom reward shaping
env = RewardShapingWrapper(env, gamma=0.99)

# Monitor training
env = Monitor(env)

# Set seed
env.reset(seed=42)

# Original PPO configuration
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    seed=42,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    )
)

print("Reward-shaped PPO training started...")

model.learn(total_timesteps=500_000)

# Save separately
model.save("models/ppo_lunar_lander_shaped")

print("\nReward-shaped PPO training completed!")
print("Model saved successfully.")

env.close()
