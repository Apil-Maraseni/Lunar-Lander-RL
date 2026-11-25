# LunarLander-v3: Reinforcement Learning Experiments

## 1. Project Objective

The objective of this project is to train an autonomous agent to land a lunar lander safely using Reinforcement Learning.

The project uses PPO (Proximal Policy Optimization) from Stable-Baselines3 and Gymnasium's LunarLander-v3 environment.

Three approaches were investigated:

1. Baseline PPO
2. Hyperparameter-tuned PPO using Optuna
3. PPO with Potential-Based Reward Shaping

## 2. Environment

- Environment: LunarLander-v3
- Library: Gymnasium
- RL Algorithm: Proximal Policy Optimization (PPO)
- Policy: MlpPolicy
- Observation space: 8-dimensional continuous vector
- Action space: 4 discrete actions
- Evaluation: Deterministic policy predictions

## 3. Baseline PPO

The baseline agent was trained using the following configuration:

| Parameter | Value |
|---|---:|
| Learning rate | 0.0003 |
| Number of steps | 2048 |
| Batch size | 64 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip range | 0.2 |
| Total timesteps | 500,000 |
| Network architecture | [64, 64] |

The trained model was evaluated using 100 episodes.

## 4. Hyperparameter Optimization

Optuna was used to investigate different PPO hyperparameter configurations.

The search included:

- Learning rate
- Discount factor (gamma)
- Number of rollout steps
- Batch size

Eight trials were conducted, each using 200,000 training timesteps.

The best-performing trial in this initial search used:

| Parameter | Value |
|---|---:|
| Learning rate | 0.00039744 |
| Gamma | 0.98572 |
| Number of steps | 4096 |
| Batch size | 64 |

This configuration was subsequently trained for 500,000 timesteps.

The evaluation showed that this tuning experiment did not outperform the baseline model.

## 5. Potential-Based Reward Shaping

Reward shaping was introduced to provide additional learning feedback to the agent.

A custom Gymnasium wrapper was implemented to calculate a shaping reward using the potential difference between consecutive states.

The shaping function was:

F(s, s') = gamma * Phi(s') - Phi(s)

The potential function considered six observation components:

- Horizontal position
- Vertical position
- Horizontal velocity
- Vertical velocity
- Angular position
- Angular velocity

The shaping reward was combined with the original environment reward:

Combined Reward = Original Reward + Shaping Reward

The discount factor used by the wrapper was 0.99, matching the PPO discount factor.

Terminal states were assigned zero potential.

The reward-shaped model was trained for 500,000 timesteps using the baseline PPO configuration.

## 6. Experimental Results

All three models were evaluated using:

- 100 episodes per model
- Identical random seeds (1000–1099)
- Original LunarLander-v3 environment
- Deterministic policy predictions

### 6.1 Performance Comparison

| Evaluation Metric | Baseline PPO | Tuned PPO | Reward-Shaped PPO |
|---|---:|---:|---:|
| Average Reward | 129.47 | 93.52 | 235.02 |
| Standard Deviation | 104.95 | 117.36 | 49.08 |
| Best Reward | 236.66 | 218.24 | 301.62 |
| Worst Reward | -100.07 | -144.02 | -4.50 |
| Positive-Reward Episodes | 79/100 | 70/100 | 99/100 |

### 6.2 Performance Improvement

Compared with baseline PPO, the reward-shaped model achieved:

- 81.5% higher average reward.
- Approximately 53.2% lower reward standard deviation.
- 99 positive-reward episodes compared with 79 for baseline PPO.

The reward-shaped model demonstrated substantially better observed performance and consistency.

## 7. Analysis and Discussion

The baseline PPO agent learned to perform the landing task but showed considerable variation across evaluation episodes.

The initial Optuna hyperparameter search did not produce a configuration that outperformed the baseline. This indicates that the tested hyperparameter combinations were not sufficient to improve performance in this experiment.

Reward shaping produced the strongest results. By providing additional feedback based on the lander's position, velocity, and orientation, the agent received guidance about desirable state transitions during training.

The reward-shaped model achieved the highest average reward and lowest standard deviation among the three approaches.

## 8. Conclusion

The experiment demonstrates that reward shaping can substantially improve the observed performance of a PPO agent in LunarLander-v3.

Among the three approaches, the reward-shaped PPO model achieved the best overall evaluation results.

The experiment also highlights that improving the reward signal can be an effective approach to reinforcement learning, beyond relying exclusively on hyperparameter optimization.

## 9. Limitations and Future Improvements

- The evaluation used 100 episodes per model; additional seeds and repeated training runs would strengthen the findings.
- Positive reward is used as a practical performance indicator, not a definitive measure of successful landing.
- The potential function uses manually selected weights that have not been individually optimized.
- Further experiments could investigate different potential functions and reward-shaping weights.
- Future work could include direct landing-success measurements and visual trajectory analysis.

