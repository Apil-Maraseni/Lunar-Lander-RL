


# we'll tune four parameters while keeping the rest of PPO unchanged. 
#learning rate, Gamma, n_steps and batch_size
import gymnasium as gym
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def evaluate_model(model, episodes=10):
    """Evaluate a model without rendering the environment."""

    env = gym.make("LunarLander-v3")
    rewards = []

    for episode in range(episodes):
        observation, info = env.reset(seed=1000 + episode)

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        rewards.append(total_reward)

    env.close()

    return sum(rewards) / len(rewards)


def objective(trial):

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 5e-4, log=True
    )

    gamma = trial.suggest_float(
        "gamma", 0.98, 0.999
    )

    n_steps = trial.suggest_categorical(
        "n_steps", [1024, 2048, 4096]
    )

    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256]
    )

    # Create training environment
    env = Monitor(gym.make("LunarLander-v3"))
    env.reset(seed=42)

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
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
        verbose=0
    )

    print(f"\nStarting Trial {trial.number}")
    print(f"Parameters: {trial.params}")

    # Train the model
    model.learn(total_timesteps=200_000)

    # Evaluate the trained model
    mean_reward = evaluate_model(model, episodes=10)

    env.close()

    print(f"Trial {trial.number} Mean Reward: {mean_reward:.2f}")

    return mean_reward


if __name__ == "__main__":

    study = optuna.create_study(
        study_name="lunar_lander_ppo",
        direction="maximize",
        storage="sqlite:///experiments/ppo_tuning.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=8)

    print("\n========== TUNING RESULTS ==========")

    print("Best Trial:", study.best_trial.number)
    print("Best Mean Reward:", study.best_value)

    print("\nBest Hyperparameters:")

    for parameter, value in study.best_params.items():
        print(f"{parameter}: {value}")
