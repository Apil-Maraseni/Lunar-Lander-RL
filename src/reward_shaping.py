
import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):

    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self.previous_potential = None

    def potential(self, observation):

        x_position = observation[0]
        y_position = observation[1]

        x_velocity = observation[2]
        y_velocity = observation[3]

        angle = observation[4]
        angular_velocity = observation[5]

        # Calculate desirability of the current state
        potential = -(
            1.0 * abs(x_position)
            + 0.5 * abs(y_position - 0.15)
            + 0.5 * abs(x_velocity)
            + 0.5 * abs(y_velocity)
            + 1.0 * abs(angle)
            + 0.25 * abs(angular_velocity)
        )

        return potential

    def reset(self, **kwargs):

        observation, info = self.env.reset(**kwargs)

        self.previous_potential = self.potential(observation)

        return observation, info

    
    def step(self, action):

        observation, reward, terminated, truncated, info = (
        self.env.step(action)
        )

    # Terminal states must have zero potential
        if terminated:
            current_potential = 0.0
        else:
            current_potential = self.potential(observation)

        shaping_reward = (
        self.gamma * current_potential
        - self.previous_potential
        )

        info["original_reward"] = reward
        info["shaping_reward"] = shaping_reward

        combined_reward = reward + shaping_reward

        self.previous_potential = current_potential

        return (
        observation,
        combined_reward,
        terminated,
        truncated,
        info
        )

