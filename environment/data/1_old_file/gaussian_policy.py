import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Gaussian2DPolicy:
    def __init__(self, mu1, sigma1, mu2, sigma2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.weight = 100
        self.bias = 300
        self.rewards = []

    def action(self):
        action1 = np.random.normal(self.mu1.numpy(), self.sigma1.numpy())
        action2 = np.random.normal(self.mu2.numpy(), self.sigma2.numpy())
        return (action1, action2)

    def update(self, actions, rewards, learning_rate=0.01):
        self.rewards.extend(rewards)
        baseline = np.mean(self.rewards)
        with tf.GradientTape() as tape:
            reward = tf.reduce_mean((rewards - baseline) * tf.math.log(tf.maximum(self.probability(actions), 1e-9)))
            # reward = tf.reduce_mean(rewards * tf.math.log(tf.maximum(self.probability(actions), 1e-6)))
            
        grads = tape.gradient(reward, [self.mu1, self.sigma1, self.mu2, self.sigma2])
        
        self.mu1.assign_add(learning_rate * grads[0])
        if self.sigma1.numpy() + learning_rate * grads[1] > 0:
            self.sigma1.assign_add(learning_rate * grads[1])
        self.mu2.assign_add(learning_rate * grads[2])
        if self.sigma2.numpy() + learning_rate * grads[3] > 0:
            self.sigma2.assign_add(learning_rate * grads[3])

    def probability(self, actions):
        prob1 = (1 / (self.sigma1 * np.sqrt(2 * np.pi))) * tf.exp(-0.5 * ((actions[:,0] - self.mu1) / self.sigma1)**2)
        prob2 = (1 / (self.sigma2 * np.sqrt(2 * np.pi))) * tf.exp(-0.5 * ((actions[:,1] - self.mu2) / self.sigma2)**2)
        return prob1 * prob2
    
    def __str__(self):
        return f"mu1: {self.mu1.numpy()*self.weight+self.bias:.2f}, mu2: {self.mu2.numpy()*self.weight+self.bias:.2f}, sigma1: {self.sigma1.numpy()*self.weight:.2f}, sigma2: {self.sigma2.numpy()*self.weight:.2f}"
        # print mu and sigma with 2 precision

# return a initialized policy based on env (600*600)
def random_init_policy(x, y, v=1.0):
    mu1, sigma1 = tf.Variable(float(x)), tf.Variable(v)  # Initialize for 1st dimension
    mu2, sigma2 = tf.Variable(float(y)), tf.Variable(v)  # Initialize for 2nd dimension
    policy = Gaussian2DPolicy(mu1, sigma1, mu2, sigma2)
    return policy

def draw_policy(policy):
    """
    Draw an ellipse for the Gaussian policy.
    """
    # Create an ellipse representing the 1-standard deviation contour
    ellipse = Ellipse(xy=(policy.mu1.numpy()*policy.weight+policy.bias, policy.mu2.numpy()*policy.weight+policy.bias), \
        width=2*policy.sigma1.numpy()*policy.weight, height=2*policy.sigma2.numpy()*policy.weight, edgecolor='r', fc='None', lw=2)
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
