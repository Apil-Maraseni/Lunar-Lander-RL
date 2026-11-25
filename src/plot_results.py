
import matplotlib.pyplot as plt
import numpy as np
import os

# Results from the 100-episode evaluation
models = ["Baseline PPO", "Tuned PPO", "Reward-Shaped PPO"]

average_rewards = [129.47, 93.52, 235.02]
std_deviations = [104.95, 117.36, 49.08]
positive_episodes = [79, 70, 99]

os.makedirs("experiments/plots", exist_ok=True)


def create_bar_chart(values, title, ylabel, filename):
    plt.figure(figsize=(9, 5))

    bars = plt.bar(models, values)

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Display values above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(f"experiments/plots/{filename}", dpi=300)
    plt.show()
    plt.close()


# 1. Average reward comparison
create_bar_chart(
    average_rewards,
    "Average Reward Comparison",
    "Average Reward",
    "average_reward.png"
)

# 2. Reward variability comparison
create_bar_chart(
    std_deviations,
    "Reward Standard Deviation Comparison",
    "Standard Deviation",
    "reward_std_deviation.png"
)

# 3. Positive-reward episode comparison
create_bar_chart(
    positive_episodes,
    "Positive-Reward Episodes (100 Evaluations)",
    "Number of Positive-Reward Episodes",
    "positive_episodes.png"
)

print("\nAll comparison graphs saved in experiments/plots/")
