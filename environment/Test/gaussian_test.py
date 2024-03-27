import numpy as np
import tensorflow as tf
from gaussian_policy_torch import *

# Example usage
policy = random_init_policy(0, 0, v=1.0)

# Example training loop
for episode in range(5000):
    actions = []
    rewards = []

    # Sample some actions and rewards
    for _ in range(1):  # Example: 10 actions per episode
        action = policy.action()
        reward = -((action[0]-(90-300)/100)**2 + (action[1]-(400-300)/100)**2)*0.1  # Define your environment_step function
        actions.append(action)
        rewards.append(reward)

    actions = np.array(actions)
    rewards = np.array(rewards)
    print(episode, str(policy))
    policy.update(actions, rewards, learning_rate=0.01)

    # if policy.sigma1 < 1e-3 or policy.sigma2 < 1e-3:
    #     break
