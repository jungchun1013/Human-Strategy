import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Gaussian2DPolicy:
    def __init__(self, mu1, sigma1, mu2, sigma2):
        self.mu1 = torch.tensor(mu1, requires_grad=True)
        self.sigma1 = torch.tensor(sigma1, requires_grad=True)
        self.mu2 = torch.tensor(mu2, requires_grad=True)
        self.sigma2 = torch.tensor(sigma2, requires_grad=True)
        self.rewards = []
        self.weight = 100
        self.bias = 300

    def action(self):
        action1 = torch.normal(self.mu1, self.sigma1)
        action2 = torch.normal(self.mu2, self.sigma2)
        return (float(action1), float(action2))

    def update(self, actions, rewards, learning_rate=0.01):
        self.rewards.extend(rewards)
        baseline = np.mean(self.rewards)
        
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        # rewards = torch.tensor(rewards) - baseline

        prob = self.probability(actions)
        reward = torch.mean(rewards * torch.log(torch.clamp(prob, min=1e-6)))

        reward.backward()

        with torch.no_grad():
            self.mu1 += learning_rate * self.mu1.grad
            self.mu2 += learning_rate * self.mu2.grad
            self.mu1.data = torch.clamp(self.mu1 + learning_rate * self.mu1.grad, min=-3, max=3)
            self.mu2.data = torch.clamp(self.mu2 + learning_rate * self.mu2.grad, min=-3, max=3)

            # Ensure sigma remains positive
            self.sigma1.data = torch.clamp(self.sigma1 + learning_rate * self.sigma1.grad, min=1e-6, max=3)
            self.sigma2.data = torch.clamp(self.sigma2 + learning_rate * self.sigma2.grad, min=1e-6, max=3)

            # Zero the gradients
            self.mu1.grad.zero_()
            self.sigma1.grad.zero_()
            self.mu2.grad.zero_()
            self.sigma2.grad.zero_()

    def probability(self, actions):
        prob1 = torch.distributions.Normal(self.mu1, self.sigma1).log_prob(actions[:, 0])
        prob2 = torch.distributions.Normal(self.mu2, self.sigma2).log_prob(actions[:, 1])
        return torch.exp(prob1 + prob2)
    
    def __str__(self):
        return f"mu1: {float(self.mu1*self.weight+self.bias):.2f}, mu2: {float(self.mu2*self.weight+self.bias):.2f}, sigma1: {float(self.sigma1*self.weight):.2f}, sigma2: {float(self.sigma2*self.weight):.2f}"
        # print mu and sigma with 2 precision

# return a initialized policy based on env (600*600)
def random_init_policy(x, y, v=1.0):
    mu1, sigma1 = float(x), v
    mu2, sigma2 = float(y), v
    policy = Gaussian2DPolicy(mu1, sigma1, mu2, sigma2)
    return policy
def draw_policy(policy):
    """
    Draw an ellipse for the Gaussian policy.
    """
    # Create an ellipse representing the 1-standard deviation contour
    ellipse = Ellipse(xy=(policy.mu1*policy.weight+policy.bias, policy.mu2*policy.weight+policy.bias), \
        width=2*policy.sigma1*policy.weight, height=2*policy.sigma2*policy.weight, edgecolor='r', fc='None', lw=2)
    # ellipse = Ellipse(xy=(policy.mu1.numpy(), policy.mu2.numpy()), width=2*policy.sigma1.numpy(), height=2*policy.sigma2.numpy(), edgecolor='r', fc='None', lw=2)
    return ellipse

def plot_policies(policies, pos):
    """
    Plot each policy as an ellipse.
    policies: List of tuples (mu1, mu2, sigma1, sigma2)
    """
    fig, ax = plt.subplots()

    for policy in policies.values():
        ellipse = draw_policy(policy)
        ax.add_patch(ellipse)
    plt.plot(pos[0], pos[1], 'go')

    # Set plot limits and labels
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_xlabel('Action Dimension 1')
    ax.set_ylabel('Action Dimension 2')
    ax.set_title('Policy Gaussian Ellipses')
    plt.savefig('plot.png')
    plt.close()
