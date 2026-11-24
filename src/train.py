
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1. Create the environment
env = gym.make("LunarLander-v3")
env = Monitor(env)

# 2. Create the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,   #controls how much the nw updates
    n_steps=2048,           #steps collected before a PPO (Proximal Policy Optimization) update
    batch_size=64,          #samples processed in one minibatch
    n_epochs=10,            #traiing passes over collected data
    gamma=0.99,             #importance of future rewards
    gae_lambda=0.95,        #controls generalized advantage estimation
    clip_range=0.2,         #restricts policy update size
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    )
)

# 3. Train the agent
print("Training started...")

model.learn(total_timesteps=500_000)   #total env steps

# 4. Save the trained model
model.save("models/ppo_lunar_lander")

print("Training completed!")
print("Model saved successfully.")

env.close()
