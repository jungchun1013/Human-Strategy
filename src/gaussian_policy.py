import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import product
from scipy.stats import multivariate_normal


    # Your code here
class Gaussian2DPolicy:
    def __init__(self, x, y, v):
        self.rewards = []
        self.params = {}
        self.scale = v
        # NOTE - param of gaussian, 2D gaussian, three tools
        for var, i, j in product(['mu', 'sigma'], range(2), range(3)):
            param_name = f"obj{j+1}_{var}{i+1}"
            if var == 'mu' and i+1 == 1:
                self.params[param_name] = torch.tensor(
                    float(x), 
                    requires_grad=True
                )
            elif var == 'mu' and i+1 == 2:
                self.params[param_name] = torch.tensor(
                    float(y), 
                    requires_grad=True
                )
            elif var == 'sigma':
                self.params[param_name] = torch.tensor(
                    float(v), 
                    requires_grad=True
                )
        # parameters of sampling tools
        self.params['prob'] = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)


    def action(self):
        prob = torch.nn.functional.softmax(self.params['prob'], dim=0)
        actiont = torch.distributions.Categorical(prob).sample()
        action1 = torch.normal(
            self.params[f"obj{actiont+1}_mu1"], self.params[f"obj{actiont+1}_sigma1"]
        )
        action2 = torch.normal(
            self.params[f"obj{actiont+1}_mu2"], self.params[f"obj{actiont+1}_sigma2"]
        )
        return (float(action1), float(action2), int(actiont))

    def update(self, actions, rewards, learning_rate=0.01):
        self.rewards.extend(rewards)
        baseline = np.mean(self.rewards)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        gaussian_learning_rate = learning_rate * 1000
        # rewards = torch.tensor(rewards) - baseline

        prob = self.probability(actions)
        reward = torch.mean(rewards * torch.log(torch.clamp(prob, min=1e-18)))
        reward.backward()
        with torch.no_grad():
            # reward needs to be params' function
            # only update current tool's gaussian params
            tool = int(actions[:, 2][0]) + 1
            for var, i in product(['mu', 'sigma'], range(2)):
                param_name = f"obj{tool}_{var}{i+1}"
                if 'mu' in param_name:
                    self.params[param_name].data = torch.clamp(self.params[param_name] + gaussian_learning_rate * self.params[param_name].grad, min=0, max=600)
                elif 'sigma' in param_name:
                    self.params[param_name].data = torch.clamp(self.params[param_name] + gaussian_learning_rate * self.params[param_name].grad, min=1)
                self.params[param_name].grad.zero_()
            self.params['prob'] += learning_rate * self.params['prob'].grad
            self.params['prob'].grad.zero_()        

    def probability(self, actions):
        prob = torch.nn.functional.softmax(self.params['prob'], dim=0)
        prob_tool = torch.distributions.Categorical(prob).log_prob(actions[:, 2])
        probs = []
        for i in range(actions.shape[0]):
            tool = int(actions[i, 2]) + 1
            prob1 = torch.distributions.Normal(
                self.params[f"obj{tool}_mu1"], self.params[f"obj{tool}_sigma1"]
            ).log_prob(actions[i, 0])
            prob2 = torch.distributions.Normal(
                self.params[f"obj{tool}_mu2"], self.params[f"obj{tool}_sigma2"]
            ).log_prob(actions[i, 1])
            probs.append(torch.exp(prob_tool[i] + prob1 + prob2))
        return torch.stack(probs)
    
    def __str__(self):
        prob = torch.nn.functional.softmax(self.params['prob'], dim=0)
        string = f"obj1: {float(prob[0]):.2f} obj2: {float(prob[1]):.2f} obj3: {float(prob[2]):.2f}\n"
        for j in range(3):
            string += f"""obj{j+1} mu1: {float(self.params[f"obj{j+1}_mu1"]):.2f}, mu2: {float(self.params[f"obj{j+1}_mu2"]):.2f}, sigma1: {float(self.params[f"obj{j+1}_sigma1"]):.2f}, sigma2: {float(self.params[f"obj{j+1}_sigma2"]):.2f}\n"""
        # print mu and sigma with 2 precision
        return string

# return a initialized policy based on env (600*600)
def initialize_policy(x, y, v=1.0):
    policy = Gaussian2DPolicy(x, y, v)
    return policy

def draw_policy(policy):
    """
    Draw an ellipse for the Gaussian policy.
    """
    # Create an ellipse representing the 1-standard deviation contour
    ellipses = []
    colors = ['r', 'g', 'b']
    for j in range(3):
        ellipses.append(Ellipse(
            xy=(policy.params[f"obj{j+1}_mu1"],
                policy.params[f"obj{j+1}_mu2"]),
            width=policy.params[f"obj{j+1}_sigma1"],
            height=policy.params[f"obj{j+1}_sigma2"],
            edgecolor=colors[j],
            fc='None',
            lw=2)
        )
    return ellipses

def plot_policies(args, policy, pos, tool, filename):
    """
    Plot each policy as an ellipse.
    """
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b']

    ellipses = draw_policy(policy)
    for e in ellipses:
        ax.add_patch(e)
    if pos is not None:
        plt.plot(pos[0], pos[1], colors[tool]+'o')
    for j in range(3):
        mu_x = policy.params[f"obj{j+1}_mu1"].detach().numpy()
        mu_y = policy.params[f"obj{j+1}_mu2"].detach().numpy()
        sigma_x = policy.params[f"obj{j+1}_sigma1"].detach().numpy()
        sigma_y = policy.params[f"obj{j+1}_sigma2"].detach().numpy()
        distr = multivariate_normal([mu_x, mu_y], [[(sigma_x**2), 0], [0, (sigma_y**2)]])

        data = distr.rvs(size = 1000)
        plt.plot(data[:,0],data[:,1], 'o', c=colors[j], alpha=0.5, markeredgewidth=0)
    # Set plot limits and labels
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Policy Gaussian Ellipses')
    plt.savefig(filename)
    plt.close()
