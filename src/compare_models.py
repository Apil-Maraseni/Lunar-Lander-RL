
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_model(model_path, model_name):
    model = PPO.load(model_path)

    env = gym.make("LunarLander-v3")

    rewards = []
    successful_landings = 0

    for episode in range(10):
        observation, info = env.reset(seed=1000 + episode)

        terminated = False
        truncated = False
        total_reward = 0
        final_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward
            final_reward = reward

        rewards.append(total_reward)

        # Positive terminal reward indicates a successful landing
        if terminated and final_reward > 0:
            successful_landings += 1

    env.close()

    average = sum(rewards) / len(rewards)

    print(f"\n{model_name}")
    print("-" * 35)
    print(f"Average Reward: {average:.2f}")
    print(f"Best Reward: {max(rewards):.2f}")
    print(f"Worst Reward: {min(rewards):.2f}")
    print(f"Successful Landings: {successful_landings}/10")

    return average


if __name__ == "__main__":

    print("Evaluating Original PPO...")
    baseline_score = evaluate_model(
        "models/ppo_lunar_lander",
        "Original PPO"
    )

    print("\nEvaluating Tuned PPO...")
    tuned_score = evaluate_model(
        "models/ppo_lunar_lander_tuned",
        "Tuned PPO"
    )

    print("\n========== FINAL COMPARISON ==========")
    print(f"Original PPO: {baseline_score:.2f}")
    print(f"Tuned PPO:    {tuned_score:.2f}")
    print(f"Difference:   {tuned_score - baseline_score:+.2f}")
