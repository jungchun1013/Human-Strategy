
import numpy as np

class UCB:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.counts = np.zeros(n_actions, dtype=np.float64)
        self.values = np.zeros(n_actions, dtype=np.float64)

    def select_action(self):
        if np.any(self.counts == 0):
            # If we have any actions that we haven't taken yet, prefer these
            action = np.random.choice(np.where(self.counts == 0)[0])
        else:
            # Calculate UCB values
            ucb_values = self.values + np.sqrt(2 * np.log(np.sum(self.counts)) / self.counts)
            action = np.argmax(ucb_values)
        return action

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] = ((self.counts[action] - 1) / self.counts[action]) * self.values[action] + reward / self.counts[action]



if __name__ == '__main__':

    # Define the true rewards for each action
    true_rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Initialize the UCB agent
    ucb_agent = UCB(n_actions=len(true_rewards))

    # Run the UCB algorithm for 1000 steps
    for step in range(1000):
        # Select an action
        action = ucb_agent.select_action()
        print(action, end= ' ')

        # Get the reward
        reward = np.random.normal(loc=true_rewards[action])

        # Update the UCB agent
        ucb_agent.update(action, reward)

        # Print the estimated values
        print(ucb_agent.values)