
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create environment
env = Monitor(gym.make("LunarLander-v3"))
env.reset(seed=42)

# Best hyperparameters from Optuna Trial 5
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.00039743821725489985,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.9857222825315675,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    ),
    seed=42,
    device="cpu",
    verbose=1
)

print("Tuned PPO training started...")

# Train for 500K timesteps
model.learn(total_timesteps=500_000)

# Save separately from the baseline
model.save("models/ppo_lunar_lander_tuned")

print("\nTuned PPO training completed!")
print("Model saved successfully.")

env.close()
